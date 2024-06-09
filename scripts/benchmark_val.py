import os.path
from pathlib import Path
from tqdm import tqdm

import torch
from torchvision import transforms
import pytorch_lightning as pl
import hydra
from PIL import Image
import time

from homo_transformer.common import setup_config, setup_network, setup_data_module
from homo_transformer.data.apolloscape_dataset import encode, get_palettedata
from homo_transformer.common import load_backbone

water_hazart = True

if water_hazart:
    CHECKPOINT_PATH = Path.cwd() / 'your model.ckpt path'
    Test_name_prefix = 'on_' 
    Test_name_suffix = '.png'
    save_dir = 'on_road'
    ori_size = (360, 1280) 
    pad_size = (1, 360, 1280)
else:
    CHECKPOINT_PATH = Path.cwd() / 'your model.ckpt path'
    Test_name_prefix = 'day_'
    Test_name_suffix = '.png'
    save_dir = 'day' 
    ori_size = (1084, 3384) 
    pad_size = (1, 1626, 3384)


def setup(cfg):

    cfg.loader.batch_size = 1

    if 'mixed_precision' not in cfg:
        cfg.mixed_precision = False

    if 'device' not in cfg:
        cfg.device = 'cuda'


@hydra.main(config_path=Path.cwd() / 'config', config_name='config.yaml')
def main(cfg):
    setup_config(cfg, setup)

    pl.seed_everything(2022, workers=True)

    data = setup_data_module(cfg)
    loader = data.val_dataloader(shuffle=False)

    device = torch.device(cfg.device)

    network = load_backbone(CHECKPOINT_PATH)
    network = network.to(device)
    network.eval()

    resize = transforms.Resize(ori_size, interpolation=transforms.InterpolationMode.NEAREST)

    i = 0
    with torch.cuda.amp.autocast(enabled=cfg.mixed_precision):
        with torch.no_grad():
            for batch in loader:

                batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}

                torch.cuda.synchronize()
                start_time = time.perf_counter()

                pred = network(batch)
                if water_hazart:
                    img = torch.sigmoid(pred['mask']) > 0.4
                    img = img.squeeze(1)
                else:
                    img = encode(pred['mask'])
                # to original size
                img = resize(img)
                img = torch.cat((torch.zeros(pad_size).to(img),img), dim=-2)

                if water_hazart:
                    img_p = transforms.functional.to_pil_image(img.squeeze(0).float(), 'L')
                    image_name = '%s%09d%s' % (Test_name_prefix, i, Test_name_suffix)
                else:
                    img_l = transforms.functional.to_pil_image(img.squeeze(0), 'L')
                    img_p = img_l.convert('P')
                    img_p.putpalette(get_palettedata())
                if not os.path.exists(save_dir):
                    os.mkdir(save_dir)

                if 'name' in batch.keys():
                    image_name = '%s%09d%s' % (Test_name_prefix, batch['name'].item(), Test_name_suffix)

                img_p.save(os.path.join(save_dir, image_name))

                i += 1


if __name__ == '__main__':
    main()
