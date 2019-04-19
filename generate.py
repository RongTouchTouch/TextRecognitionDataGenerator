import os
import random
import pandas as pd
import numpy as np
import cv2
import time

from tqdm import tqdm
from string_generator import (
    create_strings_from_dict,
    create_strings_from_file,
    create_strings_from_wikipedia,
    create_strings_randomly
)
from data_generator import FakeTextDataGenerator
from multiprocessing import Pool


def margins(margin):
    margins = margin.split(',')
    if len(margins) == 1:
        return [margins[0]] * 4
    return [int(m) for m in margins]


def load_dict(lang):
    """
    read dictionary and return all words in it
    :return:
    """
    lang_dict = []
    with open(os.path.join('dicts', lang + '.txt'),'r',encoding='utf8',errors='ignore') as d:
        lang_dict = d.readlines()
    return lang_dict


def load_files(lang):
    """
    read files and return file path to string_generator
    :return:
    """
    if lang == 'cn':
        return [os.path.join('files/cn', file) for file in os.listdir('files/cn')]
    else:
        return [os.path.join('files/latin', file) for file in os.listdir('files/latin')]


def load_fonts(lang):
    """
    read .ttf files and return the fonts
    P.S. only Truetype is allowed
    :return:
    """
    if lang == 'cn':
        return [os.path.join('fonts/cn', font) for font in os.listdir('fonts/cn')]
    else:
        return [os.path.join('fonts/latin', font) for font in os.listdir('fonts/latin')]


def gen_text_img(num, use_file, text_length, font_size, font_id, space_width, background, text_color,
                 blur, distorsion, skew_angle, thread_count):
    """

    :param num:
    :param use_file:
    :param text_length:
    :param font_size:
    :param font_id:
    :param space_width:
    :param background:
    :param text_color:
    :param blur:
    :param distorsion:
    :param skew_angle:
    :param thread_count:
    :return:
    """

    # Constant
    output_dir = 'out/'
    extension = 'jpg'
    random_skew = False
    random_blur = False
    distorsion_orientation = 0
    handwritten = False
    name_format = 0
    width = -1
    alignment = 1
    orientation = 0
    margins = (5, 5, 5, 5)
    fit = False

    language = 'cn'
    lang_dict = load_dict(language)

    fonts = load_fonts(language)

    if use_file:
        file_names = load_files(language)
        strings = create_strings_from_file(file_names, text_length, num)
    else:
        # to be implemented
        strings = create_strings_from_dict(text_length, num, lang_dict)

    p = Pool(thread_count)
    result = []
    for _, img in p.imap_unordered(
            FakeTextDataGenerator.generate_from_tuple,
            zip(
                [i for i in range(0, num)],
                strings,
                [fonts[font_id]] * num if font_id else [fonts[random.randrange(0, len(fonts))] for _ in range(0, num)],
                [output_dir] * num,
                [font_size] * num,
                [extension] * num,
                [skew_angle] * num,
                [random_skew] * num,
                [blur] * num,
                [random_blur] * num,
                [background] * num,
                [distorsion] * num,
                [distorsion_orientation] * num,
                [handwritten] * num,
                [name_format] * num,
                [width] * num,
                [alignment] * num,
                [text_color] * num,
                [orientation] * num,
                [space_width] * num,
                [margins] * num,
                [fit] * num
            )
    ):
        result.append((_, img))

    p.terminate()
    target = np.concatenate([img for _, img in result], axis=1)
    df = pd.concat([meta for meta, _ in result])

    return df, target


if __name__ == '__main__':
    num = 10
    use_file = 1
    text_length = 10
    font_size = 40
    font_id = 1
    space_width = 1
    background = 0
    text_color = '#282828'
    blur = 0
    distorsion = 0
    skew_angle = 1
    thread_count = 8

    start_time = time.time()
    df, target = gen_text_img(num, use_file, text_length, font_size, font_id, space_width, background, text_color,
                              blur, distorsion, skew_angle, thread_count)
    cv2.imwrite(os.path.join('out/' + 'target.jpg'), target)

    end_time = time.time()
    print('time for synthesize 10 image:', end_time - start_time)
