import logging
from os import path
from typing import Iterable
from itertools import chain
import numpy as np
import torch
from torch.optim import Adam
import torch.nn as nn

from models.densenet_ import DenseNet
from generate import gen_text_img
from utils.converter import LabelConverter

logger = logging.getLogger(__name__)
__pwd__ = path.realpath(path.dirname(__file__))


class TextEncoder(object):
    def __init__(self, alphabet: Iterable):
        self.char2int = {c:i for i, c in enumerate(sorted(list(set(alphabet))))}
        logging.info(f'encoder have {len(self.char2int)} unique alphabet')
        self.unknown_index = len(self.char2int)

    def __call__(self, chars: Iterable):
        return [self.char2int.get(c, self.unknown_index) for c in chars]

    def size(self):
        return len(self.char2int)+1


def train():
    # create char dict
    with open(path.join(__pwd__, 'dicts', 'chars.txt'), 'rt') as fp:
        encoder = TextEncoder(chain.from_iterable(l.strip('\n') for l in fp.readlines()))

    # create DenseNet model
    """
    model_params = {}
    model_params['architecture'] = 'densenet121'
    model_params['num_classes'] = len(encoder) + 1
    model_params['mean'] = (0.5,)
    model_params['std'] = (0.5,)
    model = init_network(model_params)
    """
    model = DenseNet(img_height=32, drop_rate=0.2, num_classes=encoder.size())


    logger.info('parameters:', sum(t.numel() for t in model.parameters() if t.requires_grad))

    criterion = nn.CTCLoss()
    # criterion = nn.CTCLoss(zero_infinity=True)
    # define optimizer
    optimizer = Adam(p for p in model.parameters() if p.requires_grad)

    converter = LabelConverter(encoder, ignore_case=False)

    for batch in range(100):
        num = 10
        use_file = 1
        text_length = -1
        font_size = 0
        font_id = 1
        space_width = 1
        text_color = '#282828'
        thread_count = 8

        random_skew = True
        skew_angle = 3
        random_blur = True
        blur = 1

        distorsion = 0
        background = 1

        text_meta, text_img = gen_text_img(num, use_file, text_length, font_size, font_id, space_width, background, text_color,
                                  blur, random_blur, distorsion, skew_angle, random_skew, thread_count)
        # convert it into gray
        text_img = np.array(text_img[:,:,1]).astype(np.float32) / 255.0 - 0.5

        # create pytorch input and output
        X = torch.from_numpy(text_img.reshape([1, 1, 32, -1]))
        Y = torch.LongTensor(list(map(encoder, text_meta['text'])))

        # forward
        logit = model.forward(X)
        logging.debug(logit.shape)
        # loss = nn.CTCLoss(X, Y)

        #backward
        # loss.backward()

        # optimizer.step()


if __name__ == '__main__':
    train()