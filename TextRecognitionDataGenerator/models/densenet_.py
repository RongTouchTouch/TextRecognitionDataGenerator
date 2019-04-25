import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

class _ConvBlock(nn.Sequential):
    def __init__(self, num_input_features, growth_rate, drop_rate):
        """BN+RL+CONV
        :return output the same WxH
        """
        super().__init__()
        self.add_module('norm', nn.BatchNorm2d(num_input_features, eps=1.1e-5)),
        self.add_module('relu', nn.ReLU(inplace=True)),
        self.add_module('conv', nn.Conv2d(num_input_features, growth_rate,
                        kernel_size=3, stride=1, padding=1, bias=True))
        self.drop_rate = drop_rate

    def forward(self, x):
        new_features = super().forward(x)
        if self.drop_rate > 0:
            new_features = F.dropout(new_features, p=self.drop_rate, training=self.training)
        return new_features

class _DenseBlock(nn.ModuleList):
    def __init__(self, num_layers, num_input_features, growth_rate, drop_rate):
        super(_DenseBlock, self).__init__()
        for i in range(num_layers):
            """there is no bottle neck in each DenseLayer"""
            layer = _ConvBlock(num_input_features + i * growth_rate, growth_rate, drop_rate)
            self.add_module('denselayer%d' % (i + 1), layer)

    def forward(self, x):
        for l in self.children():
            x = torch.cat([x, l(x)], 1)
        return x

class _Transition(nn.Sequential):
    def __init__(self, num_input_features, num_output_features):
        super(_Transition, self).__init__()
        self.add_module('norm', nn.BatchNorm2d(num_input_features, eps=1.1e-5))
        self.add_module('relu', nn.ReLU(inplace=True))
        self.add_module('conv', nn.Conv2d(num_input_features, num_output_features,
                                          kernel_size=1, stride=1, bias=False, ))
        ## there should be a L2 regularizer for the conv here
        self.add_module('pool', nn.AvgPool2d(kernel_size=2, stride=2))

class DenseNet(nn.Module):
    def __init__(self, img_height=32, growth_rate=8,
                 block_config=(8, 8, 8),
                 compression_config=(1.0, 2/3, None),
                 num_init_features=64, drop_rate=0.2, num_classes=1000):
        """
        Densenet for OCR

        :param growth_rate: (int) - how many filters to add each layer (`k` in paper)
        :param block_config: (list of 3 ints) - how many layers in each pooling block
        :param compression_config: (list of 3 float in (0,1]) - the compression level in each pooling block
        :param num_init_features: (int) - the number of filters to learn in the first convolution layer
        :param drop_rate: dropout rate after each dense layer
        :param num_classes: number of classification classes
        """

        super(DenseNet, self).__init__()

        # First convolution
        """same padding: floor( (L+2P-D(K-1)-1)/S +1 ) == ceil(L/S)"""
        self.features = nn.Sequential(
            nn.Conv2d(1, num_init_features, kernel_size=5, stride=2, padding=2, bias=False)
        )
        img_height = np.ceil(img_height/2)

        # Each denseblock
        num_features = num_init_features
        for i, (num_layers, compression_level) in enumerate(zip(block_config, compression_config)):
            block = _DenseBlock(num_layers=num_layers, num_input_features=num_features,
                                growth_rate=growth_rate, drop_rate=drop_rate)
            self.features.add_module('denseblock%d' % (i + 1), block)
            num_features = num_features + num_layers * growth_rate
            if compression_level:
                trans = _Transition(num_input_features=num_features,
                                    num_output_features=round(num_features*compression_level))
                self.features.add_module('transition%d' % (i + 1), trans)
                num_features = round(num_features*compression_level)
                img_height = np.floor(((img_height-2)/2)+1) # see the AvgPool2d in _Transition

        # Final batch norm
        self.features.add_module('last_norm', nn.BatchNorm2d(num_features, eps=1.1e-5))

        # Linear layer
        self.classifier = nn.Linear(num_features*int(img_height), num_classes)

    def forward(self, x: torch.Tensor):
        # x = F.pad(x, (1,2,1,2)) # this is how keras "same" padding in Conv2d
        features = self.features(x)
        out = F.relu(features, inplace=True)
        batch, channel, height, width = out.shape
        out = out.permute(0, 3, 1, 2).reshape(batch*width, -1)
        out = self.classifier(out).reshape(batch, width, -1)
        return out

if __name__ == '__main__':

    model = DenseNet(img_height=32, growth_rate=8,
                     block_config=(8,8,8,), compression_config=(1., 2/3., None),
                     num_init_features=64, drop_rate=0.2, num_classes=1000)
    print(model)

    print('parameters:', sum(t.numel() for t in model.parameters() if t.requires_grad))

    from PIL import Image, ImageOps
    with Image.open('/tmp/xx/test_line.png') as img:
        img = img.convert('L')
        width, height = img.size[0], img.size[1]
        scale = height * 1.0 / 32
        width = int(width / scale)

        img = img.resize([width, 32], Image.ANTIALIAS)

        img_array = np.array(img.convert('1'))
        boundary_array = np.concatenate((img_array[0, :], img_array[:, width - 1], img_array[31, :], img_array[:, 0]),
                                        axis=0)
        if np.median(boundary_array) == 0:  # 将黑底白字转换为白底黑字
            img = ImageOps.invert(img)

        img = np.array(img).astype(np.float32) / 255.0 - 0.5
        print(img.shape)

        X = img.reshape([1, 1, 32, width])

        logit = model.forward(torch.from_numpy(X))

        print(logit.shape)
