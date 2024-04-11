import os.path
from pathlib import Path
from tqdm import tqdm

import torch
from torchvision import transforms
import pytorch_lightning as pl
import hydra
from PIL import Image

from homo_transformer.common import setup_config, setup_network, setup_data_module
from homo_transformer.data.apolloscape_dataset import encode, get_palettedata
from cross_view_transformer.common import load_backbone

Test_name_prefix = 'day_'
Test_name_suffix = '.png'
save_dir = 'predict'
ori_size = (1084, 3384)
pad_size = (1, 1626, 3384)
CHECKPOINT_PATH = Path.cwd() / 'logs/your_dir/checkpoints/model.ckpt'

def setup(cfg):
    print('Benchmark mixed precision by adding +mixed_precision=True')
    print('Benchmark cpu performance +device=cpu')

    cfg.loader.batch_size = 1

    if 'mixed_precision' not in cfg:
        cfg.mixed_precision = False

    if 'device' not in cfg:
        cfg.device = 'cuda'


@hydra.main(config_path=Path.cwd() / 'config', config_name='config.yaml')
def main(cfg):
    setup_config(cfg, setup)

    pl.seed_everything(2022, workers=True)

    #network = setup_network(cfg)
    data = setup_data_module(cfg)
    loader = data.val_dataloader(shuffle=False)

    device = torch.device(cfg.device)

    network = load_backbone(CHECKPOINT_PATH)
    network = network.to(device)
    network.eval()

    # sample = next(iter(loader))
    # batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in sample.items()}

    resize = transforms.Resize(ori_size, interpolation=transforms.InterpolationMode.NEAREST)

    with torch.cuda.amp.autocast(enabled=cfg.mixed_precision):
        with torch.no_grad():
            for batch in loader:
                batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}

                pred = network(batch)
                img = encode(pred['mask'])
                # to original size
                img = resize(img)
                img = torch.cat((torch.zeros(pad_size).to(img),img), dim=-2)

                img_l = transforms.functional.to_pil_image(img.squeeze(0), 'L')
                img_p = img_l.convert('P')
                img_p.putpalette(get_palettedata())
                if not os.path.exists(save_dir):
                    os.mkdir(save_dir)

                if 'name' in batch.keys():
                    image_name = '%s%09d%s' % (Test_name_prefix, batch['name'].item(), Test_name_suffix)

                img_p.save(os.path.join(save_dir, image_name))


if __name__ == '__main__':
    main()
