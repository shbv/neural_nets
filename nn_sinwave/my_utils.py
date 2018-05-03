## Util functions

import numpy as np
from scipy.misc import imsave

# Crop fraction of image
def imcrop(img, fraction):
    if fraction <= 0 or fraction >= 1:
        return img
    row_i = int(img.shape[0] * fraction) // 2
    col_i = int(img.shape[1] * fraction) // 2
    return img[row_i:-row_i, col_i:-col_i]


# Create collage of all_imgs 
# Inspired from Creative Applications of Deep Learning w/ Tensorflow code
def all_imgs(images, filen):

    num_collage_rows = int(np.ceil(np.sqrt(images.shape[0])))
    num_collage_cols = num_collage_rows

    num_imgs = images.shape[0]
    img_rows = images.shape[1]
    img_cols = images.shape[2]

    if len(images.shape) == 4:
        num_channels = images.shape[3]
        matrix = np.ones((img_rows * num_collage_rows + num_collage_rows + 1, img_cols * num_collage_cols + num_collage_cols + 1, num_channels)) * 0.33
    elif len(images.shape) == 3:
        matrix = np.ones((img_rows * num_collage_rows + num_collage_rows + 1, img_cols * num_collage_cols + num_collage_cols + 1)) * 0.33
    else:
        raise ValueError('Could not parse image shape of {}'.format(images.shape))

    for i in range(num_collage_rows):
        for j in range(num_collage_cols):

            img_num = i * num_collage_rows + j

            if img_num < num_imgs:
                img = images[img_num]
                matrix[1 + i + i * img_rows:1 + i + (i + 1) * img_rows,
                       1 + j + j * img_cols:1 + j + (j + 1) * img_cols] = img

    imsave(arr=np.squeeze(matrix), name=filen)

    return matrix


