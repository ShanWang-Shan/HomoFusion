import torch
import torch.nn as nn
import torchvision

class ResNetExtractor(nn.Module):
    """
    Helper wrapper that uses torch.utils.checkpoint.checkpoint to save memory while training.

    This runs a fake input with shape (1, 3, input_height, input_width)
    to give the shapes of the features requested.

    Sample usage:
        backbone = EfficientNetExtractor(224, 480, ['reduction_2', 'reduction_4'])

        # [[1, 56, 28, 60], [1, 272, 7, 15]]
        backbone.output_shapes

        # [f1, f2], where f1 is 'reduction_1', which is shape [b, d, 128, 128]
        backbone(x)
    """
    def __init__(self, layer_names, image_height, image_width, model_name='resnet-101'):
        super().__init__()

        # We can set memory efficient swish to false since we're using checkpointing
        net = torchvision.models.resnet101(pretrained=True)
        features = nn.ModuleList(net.children())[:7] # max to ??
        self.layers = nn.Sequential(*features)

        self.idx_pick = [3,6]

        # Pass a dummy tensor to precompute intermediate shapes
        dummy = torch.rand(1, 3, image_height, image_width)
        output_shapes = [x.shape for x in self(dummy)]

        self.output_shapes = output_shapes

    def forward(self, x):
        if self.training:
            x = x.requires_grad_(True)

        result = []

        for layer in self.layers:
            if self.training:
                x = torch.utils.checkpoint.checkpoint(layer, x)
            else:
                x = layer(x)

            result.append(x)

        return [result[i] for i in self.idx_pick]


# def conv3x3(in_planes, out_planes, stride=1):
#     """3x3 convolution with padding"""
#     return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
#                      padding=1, bias=False)


# class BasicBlock(nn.Module):
#     expansion = 1
#
#     def __init__(self, inplanes, planes, stride=1, downsample=None):
#         super(BasicBlock, self).__init__()
#         self.conv1 = conv3x3(inplanes, planes, stride)
#         self.bn1 = nn.BatchNorm2d(planes)
#         self.relu = nn.ReLU(inplace=False)
#         self.conv2 = conv3x3(planes, planes)
#         self.bn2 = nn.BatchNorm2d(planes)
#         self.downsample = downsample
#         self.stride = stride
#
#     def forward(self, x):
#         residual = x
#
#         out = self.conv1(x)
#         out = self.bn1(out)
#         out = self.relu(out)
#
#         out = self.conv2(out)
#         out = self.bn2(out)
#
#         if self.downsample is not None:
#             residual = self.downsample(x)
#
#         out = out + residual
#         out = self.relu(out)
#
#         return out

#
# class Bottleneck(nn.Module):
#     expansion = 4
#
#     def __init__(self, inplanes, planes, stride=1, downsample=None):
#         super(Bottleneck, self).__init__()
#         self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
#         self.bn1 = nn.BatchNorm2d(planes)
#         self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
#                                padding=1, bias=False)
#         self.bn2 = nn.BatchNorm2d(planes)
#         self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
#         self.bn3 = nn.BatchNorm2d(planes * 4)
#         self.relu = nn.ReLU(inplace=False)
#         self.downsample = downsample
#         self.stride = stride
#
#     def forward(self, x):
#         residual = x
#
#         out = self.conv1(x)
#         out = self.bn1(out)
#         out = self.relu(out)
#
#         out = self.conv2(out)
#         out = self.bn2(out)
#         out = self.relu(out)
#
#         out = self.conv3(out)
#         out = self.bn3(out)
#
#         if self.downsample is not None:
#             residual = self.downsample(x)
#
#         out = out + residual
#         out = self.relu(out)
#
#         return out
#
#
# class B2_ResNet(nn.Module):
#     # ResNet50 with two branches
#     def __init__(self):
#         # self.inplanes = 128
#         self.inplanes = 64
#         super(B2_ResNet, self).__init__()
#
#         self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
#                                bias=False)
#         self.bn1 = nn.BatchNorm2d(64)
#         self.relu = nn.ReLU(inplace=False)
#         self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
#         self.layer1 = self._make_layer(Bottleneck, 64, 3)
#         self.layer2 = self._make_layer(Bottleneck, 128, 4, stride=2)
#         self.layer3_1 = self._make_layer(Bottleneck, 256, 6, stride=2)
#         self.layer4_1 = self._make_layer(Bottleneck, 512, 3, stride=2)
#
#         # self.inplanes = 512
#         # self.layer3_2 = self._make_layer(Bottleneck, 256, 6, stride=2)
#         # self.layer4_2 = self._make_layer(Bottleneck, 512, 3, stride=2)
#
#         for m in self.modules():
#             if isinstance(m, nn.Conv2d):
#                 n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
#                 m.weight.data.normal_(0, math.sqrt(2. / n))
#             elif isinstance(m, nn.BatchNorm2d):
#                 m.weight.data.fill_(1)
#                 m.bias.data.zero_()
#
#     def _make_layer(self, block, planes, blocks, stride=1):
#         downsample = None
#         if stride != 1 or self.inplanes != planes * block.expansion:
#             downsample = nn.Sequential(
#                 nn.Conv2d(self.inplanes, planes * block.expansion,
#                           kernel_size=1, stride=stride, bias=False),
#                 nn.BatchNorm2d(planes * block.expansion),
#             )
#
#         layers = []
#         layers.append(block(self.inplanes, planes, stride, downsample))
#         self.inplanes = planes * block.expansion
#         for i in range(1, blocks):
#             layers.append(block(self.inplanes, planes))
#
#         return nn.Sequential(*layers)
#
#     def forward(self, x):
#         x = self.conv1(x)
#         x = self.bn1(x)
#         x = self.relu(x)
#         x = self.maxpool(x)
#
#         x = self.layer1(x)
#         x = self.layer2(x)
#         x1 = self.layer3_1(x)
#         x1 = self.layer4_1(x1)
#
#         # x2 = self.layer3_2(x)
#         # x2 = self.layer4_2(x2)
#
#         return x1
#
# class Classifier_Module(nn.Module):
#     def __init__(self, dilation_series, padding_series, NoLabels, input_channel):
#         super(Classifier_Module, self).__init__()
#         self.conv2d_list = nn.ModuleList()
#         for dilation, padding in zip(dilation_series, padding_series):
#             self.conv2d_list.append(nn.Conv2d(input_channel, NoLabels, kernel_size=3, stride=1,
#                                                   padding=padding, dilation=dilation, bias=True))
#         for m in self.conv2d_list:
#             m.weight.data.normal_(0, 0.01)
#
#     def forward(self, x):
#         out = self.conv2d_list[0](x)
#         for i in range(len(self.conv2d_list) - 1):
#             out = out + self.conv2d_list[i + 1](x)
#         return out
#
# class ResnetExtractor(torch.nn.Module):
#     """
#     Helper wrapper that uses torch.utils.checkpoint.checkpoint to save memory while training.
#
#     This runs a fake input with shape (1, 3, input_height, input_width)
#     to give the shapes of the features requested.
#
#     Sample usage:
#         backbone = ResnetExtractor(224, 480, ['reduction_2', 'reduction_4', 'reduction_8'])
#
#         # [[1, 56, 112, 240], [1, 272, 7, 15]]
#         backbone.output_shapes
#
#         # [f1, f2], where f1 is 'reduction_1', which is shape [b, d, 128, 128]
#         backbone(x)
#     """
#     def __init__(self, layer_names, image_height, image_width, model_name='resnet-50'):
#         super().__init__()
#
#         self.resnet = B2_ResNet()
#
#         self.conv4 = self._make_pred_layer(Classifier_Module, [3, 6, 12, 18], [3, 6, 12, 18], 128, 2048)
#         self.conv3 = self._make_pred_layer(Classifier_Module, [3, 6, 12, 18], [3, 6, 12, 18], 128, 1024)
#         self.conv2 = self._make_pred_layer(Classifier_Module, [3, 6, 12, 18], [3, 6, 12, 18], 128, 512)
#         self.conv1 = self._make_pred_layer(Classifier_Module, [3, 6, 12, 18], [3, 6, 12, 18], 128, 256)
#
#         # Pass a dummy tensor to precompute intermediate shapes
#         dummy = torch.rand(1, 3, image_height, image_width)
#         output_shapes = [x.shape for x in self(dummy)]
#
#         self.output_shapes = output_shapes
#
#         if self.training:
#             self.initialize_weights()
#
#     def _make_pred_layer(self, block, dilation_series, padding_series, out_channel, input_channel):
#         return block(dilation_series, padding_series, out_channel, input_channel)
#
#     def forward(self, x):
#         if self.training:
#             x = x.requires_grad_(True)
#
#         x = self.resnet.conv1(x)
#         x = self.resnet.bn1(x)
#         x = self.resnet.relu(x)
#         x = self.resnet.maxpool(x)
#         x1 = self.resnet.layer1(x)  # 256 x /2 x /2
#         x2 = self.resnet.layer2(x1)  # 512 x /4 x /4
#         x3 = self.resnet.layer3_1(x2)  # 1024 x /8 x /8
#         x4 = self.resnet.layer4_1(x3)  # 2048 x /16 x /16
#
#         result = []
#         conv1_feat = self.conv1(x1)
#         result.append(conv1_feat)
#
#         conv2_feat = self.conv2(x2)
#         result.append(conv2_feat)
#
#         conv3_feat = self.conv3(x3)
#         result.append(conv3_feat)
#
#         conv4_feat = self.conv4(x4)
#         result.append(conv4_feat)
#
#         return result
#
#     def initialize_weights(self):
#         res50 = models.resnet50(pretrained=True)
#         pretrained_dict = res50.state_dict()
#         all_params = {}
#         for k, v in self.resnet.state_dict().items():
#             if k in pretrained_dict.keys():
#                 v = pretrained_dict[k]
#                 all_params[k] = v
#             elif '_1' in k:
#                 name = k.split('_1')[0] + k.split('_1')[1]
#                 v = pretrained_dict[name]
#                 all_params[k] = v
#             elif '_2' in k:
#                 name = k.split('_2')[0] + k.split('_2')[1]
#                 v = pretrained_dict[name]
#                 all_params[k] = v
#         assert len(all_params.keys()) == len(self.resnet.state_dict().keys())
#         self.resnet.load_state_dict(all_params)
