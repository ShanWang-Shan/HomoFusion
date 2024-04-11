import torch
import torch.nn as nn

from transformers import SegformerModel
# import torch
# from datasets import load_dataset


class MitExtractor(torch.nn.Module):
    """
    Helper wrapper that uses torch.utils.checkpoint.checkpoint to save memory while training.

    This runs a fake input with shape (1, 3, input_height, input_width)
    to give the shapes of the features requested.

    Sample usage:
        backbone = MitExtractor(224, 480, ['reduction_2', 'reduction_4'])

        # [[1, 56, 28, 60], [1, 272, 7, 15]]
        backbone.output_shapes

        # [f1, f2], where f1 is 'reduction_1', which is shape [b, d, 128, 128]
        backbone(x)
    """
    def __init__(self, layer_names, image_height, image_width, model_name='mit-b1'):
        super().__init__()

        idx_max = -1
        layer_to_idx = {}

        # We can set memory efficient swish to false since we're using checkpointing
        self.net = SegformerModel.from_pretrained("nvidia/"+model_name)

        self.idx_pick = [0, 2]

        # Pass a dummy tensor to precompute intermediate shapes
        dummy = torch.rand(1, 3, image_height, image_width)
        output_shapes = [x.shape for x in self(dummy)]

        self.output_shapes = output_shapes

    def forward(self, x):
        if self.training:
            x = x.requires_grad_(True)

        result = self.net(x, output_hidden_states=True)

        return [result.hidden_states[i] for i in self.idx_pick]
