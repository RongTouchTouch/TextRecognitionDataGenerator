import os
import random
import numpy as np
import pandas as pd
import platform
import torch

from PIL import Image, ImageFilter
import cv2

import computer_text_generator
import background_generator
import distorsion_generator
import image_process

class FakeTextDataGenerator(object):
    @classmethod
    def generate_from_tuple(cls, t):
        """
            Same as generate, but takes all parameters as one tuple
        """

        return cls.generate(*t)


    @classmethod
    def generate(cls, index, text, font, out_dir, size, extension, channel,
                 skewing_angle, random_skew, blur, random_blur,
                 background_type, distorsion_type, distorsion_orientation,
                 random_process, noise, erode, dilate, incline, 
                 is_handwritten, name_format, width, alignment, text_color,
                 orientation, space_width, margins, fit):

        image = None

        margin_top, margin_left, margin_bottom, margin_right = margins
        horizontal_margin = margin_left + margin_right
        vertical_margin = margin_top + margin_bottom

        ##########################
        # Create picture of text #
        ##########################
        if is_handwritten:
            if orientation == 1:
                raise ValueError("Vertical handwritten text is unavailable")
            image = handwritten_text_generator.generate(text, text_color, fit)
        else:
            image = computer_text_generator.generate(text, font, text_color, size, orientation, space_width, fit)

        random_angle = random.random()*random.randint(0-skewing_angle, skewing_angle)
        skewing_angle = skewing_angle if not random_skew else random_angle
        rotated_img = image.rotate(skewing_angle, expand=1)

        #############################
        # Apply distorsion to image #
        #############################

        if distorsion_type == 0:
            distorted_img = rotated_img # Mind = blown
        elif distorsion_type == 1:
            distorted_img = distorsion_generator.sin(
                rotated_img,
                vertical=(distorsion_orientation == 0 or distorsion_orientation == 2),
                horizontal=(distorsion_orientation == 1 or distorsion_orientation == 2)
            )
        elif distorsion_type == 2:
            distorted_img = distorsion_generator.cos(
                rotated_img,
                vertical=(distorsion_orientation == 0 or distorsion_orientation == 2),
                horizontal=(distorsion_orientation == 1 or distorsion_orientation == 2)
            )
        else:
            distorted_img = distorsion_generator.random(
                rotated_img,
                vertical=(distorsion_orientation == 0 or distorsion_orientation == 2),
                horizontal=(distorsion_orientation == 1 or distorsion_orientation == 2)
            )

        ##################################
        # Resize image to desired format #
        ##################################

        # Horizontal text
        if orientation == 0:
            new_width = int(distorted_img.size[0] * (float(size - vertical_margin) / float(distorted_img.size[1])))
            resized_img = distorted_img.resize((new_width, size - vertical_margin), Image.ANTIALIAS)
            background_width = width if width > 0 else new_width + horizontal_margin
            background_height = size
        # Vertical text
        elif orientation == 1:
            new_height = int(float(distorted_img.size[1]) * (float(size - horizontal_margin) / float(distorted_img.size[0])))
            resized_img = distorted_img.resize((size - horizontal_margin, new_height), Image.ANTIALIAS)
            background_width = size
            background_height = new_height + vertical_margin
        else:
            raise ValueError("Invalid orientation")

        #############################
        # Generate background image #
        #############################
        
        if background_type == 0:
            background = background_generator.gaussian_noise(background_height, background_width)
        elif background_type == 1:
            background = background_generator.plain_white(background_height, background_width)
        elif background_type == 2:
            background = background_generator.quasicrystal(background_height, background_width)
        else:
            background = background_generator.picture(background_height, background_width)

        #############################
        # Place text with alignment #
        #############################
        
        new_text_width, _ = resized_img.size

        if alignment == 0 or width == -1:
            background.paste(resized_img, (margin_left, margin_top), resized_img)
        elif alignment == 1:
            background.paste(resized_img, (int(background_width / 2 - new_text_width / 2), margin_top), resized_img)
        else:
            background.paste(resized_img, (background_width - new_text_width - margin_right, margin_top), resized_img)

        ##################################
        # Apply gaussian blur #
        ##################################
        
        blur = blur if not random_blur else random.random()*blur
        final_image = background.filter(
            ImageFilter.GaussianBlur(
                radius = blur
            )
        )
               
        ##################################
        # Apply image process #
        ##################################
        
        final_image = np.asarray(final_image)
        incline = min(incline, int(final_image.shape[1]/4))
        final_image, noise, dilate, erode, incline = image_process.do(random_process, noise, erode, dilate, incline, final_image)
        final_image = Image.fromarray(final_image)
        
        #####################################
        # Generate name for resulting image #
        #####################################
        if name_format == 0:
            image_name = '{}_{}.{}'.format(text, str(index), extension)
        elif name_format == 1:
            image_name = '{}_{}.{}'.format(str(index), text, extension)
        elif name_format == 2:
            image_name = '{}.{}'.format(str(index),extension)
        else:
            print('{} is not a valid name format. Using default.'.format(name_format))
            image_name = '{}_{}.{}'.format(text, str(index), extension)

        # Save the image
        if channel == 3:
            final_image = final_image.convert('RGB')
        elif channel ==1:
            final_image = final_image.convert('L')
        #final_image.save(os.path.join(out_dir, image_name))

        # Resize
        final_image = np.asarray(final_image)
        if orientation == 0:
            final_image = cv2.resize(final_image, (int(final_image.shape[1] * 32 / final_image.shape[0]), 32))
        else:
            final_image = cv2.resize(final_image, (32, int(final_image.shape[0] * 32 / final_image.shape[1])))

        if platform.system() == "Windows":
            font = font.split('\\')[1].split('.')[0]
        else:  # linux
            font = font.split('/')[2].split('.')[0]
                          
        background = ['Guassian', 'Plain white', 'Quasicrystal','Image']
        distorsion = ['None', 'Sine wave', 'Cosine wave']
        
        data = pd.DataFrame(np.array([[index, text, len(text), final_image.shape[1], size, font, skewing_angle, blur,
                                       distorsion[distorsion_type], background[background_type],
                                       noise, erode, dilate, incline]]),
                            columns=['index', 'text', 'text_length', 'img_shape', 'font_size', 'font_id', 'skew_angle',
                                     'blur', 'distorsion_type', 'background_type','num_noise','erode','dilate','incline'])
        if channel == 1 :
            final_image = np.asarray(torch.Tensor(np.asarray(final_image)).unsqueeze(0).permute(1,2,0),dtype='uint8')
        return data, final_image
