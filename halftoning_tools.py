# This module contains functions to perform halftoning on images.
#   - mask halftoning with random distribution
#   - mask with defined pattern
#       + linear filling
#       + spiral filling
#       + random filling
#   - eventually error diffusion but it's not really necessary


import cv2
import numpy as np
import matplotlib.pyplot as plt

def fun_halftone_image_with_mask(image_continuous_tone, mask_size = 4,
                                    mask_type = "random",
                                    cell_factor = 1):
    """The function takes an image as input, single channel ideally.
    In:
        - mask_size (int): defines the size of the dot mask_size x mask_size
        - mask_type (str): "random", "linear", "spiral" defines how the cell is
        filled
        - cell_factor (int): defines how big should be the cell
    Out:
        It returns an image in its halftone version.

    """
    size_image = np.shape(image_continuous_tone)

    # create the mask
    if len(size_image) > 2:
        print "The image image given is multi-channel and we don't do that here."
        image_half_tone = image_continuous_tone
    else:
        # create mask
        if mask_size > 16:
            mask_size = 16

        max_level =  np.power(mask_size,2)
        print max_level

        if mask_type == "random":
            mask = np.random.randint(0, high = max_level, # it can be any limit actually
                                    size = np.power(mask_size,2),
                                    dtype = np.uint8)
            mask_single = np.argsort(mask)
            mask = mask_single.reshape((mask_size, mask_size))
            print mask

        elif mask_type == "linear":
            mask = np.arange(max_level).reshape(mask_size,mask_size)

        elif mask_type == "spiral":
            mask = spiral_list_to_array(mask_size).reshape(mask_size,mask_size)
            # to invert the spiral
            mask = np.abs(mask - max_level + 1)

        else:
            print "We have a problem Houston."

        print "You choose a mask type [%s] of size [%1.0fx%1.0f]." % (mask_type, mask_size, mask_size)
        # image quantification to have value between 0 and the maximum the
        # mask can produce
        image_continuous_tone = (image_continuous_tone.astype(np.float32) / 256 ) \
                                    * max_level

        # Apply the cell factor
        mask = cv2.resize(mask,(mask_size * cell_factor,
                                mask_size * cell_factor),
                               interpolation = cv2.INTER_NEAREST)

        # create a mask of the image size by replicating the original cell
        factor_width  = int(np.ceil(np.shape(image_continuous_tone)[0] / mask_size)+1)
        factor_height = int(np.ceil(np.shape(image_continuous_tone)[1] / mask_size)+1)
        # --> the +1 is for safety, it will to large but I take care of that now

        # replicate the mask as a tile image
        mask_image = np.tile(mask,(factor_width, factor_height))
        mask_image = mask_image[0:np.shape(image_continuous_tone)[0],
                                0:np.shape(image_continuous_tone)[1]]

        # apply mask to coninuous image
        image_half_tone = image_continuous_tone.astype(np.uint8) >= mask_image
        image_half_tone = 255 * image_half_tone.astype(np.uint8)

        return image_half_tone

def fun_halftone_image_with_global_mask(image_continuous_tone):
    """The function takes an image as input, single channel ideally.
    It's a global mask because the mask as the image size.
    It returns an image in its halftone version.
    """
    size_image = np.shape(image_continuous_tone)

    if len(size_image) > 2:
        print "The image image given is multi-channel and we don't do that here."
        image_half_tone = image_continuous_tone
    else:
        # create mask
        mask = np.random.randint(0, high = 256,
                                 size = (np.shape(image_continuous_tone)[0],
                                            np.shape(image_continuous_tone)[1]),
                                 dtype = np.uint8)
        # apply mask to coninuous image
        image_half_tone = image_continuous_tone > mask
        image_half_tone = image_half_tone.astype(np.uint8)

        return image_half_tone

def display_image(image_data, image_name_for_title):
    """Thte function display an image, panorama image of little planet.
    """
    fig = plt.figure(figsize=(12,12))
    if len(np.shape(image_data)) > 2:
        plt.imshow(image_data)
    else:
        plt.imshow(image_data, cmap= "gray")
    plt.xticks([])
    plt.yticks([])
    plt.title(image_name_for_title)
    plt.show()

def spiral(n):
    dx,dy = 1,0            # Starting increments
    x,y = 0,0              # Starting location
    myarray = [[None]* n for j in range(n)]
    for i in xrange(n**2):
        myarray[x][y] = i
        nx,ny = x+dx, y+dy
        if 0<=nx<n and 0<=ny<n and myarray[nx][ny] == None:
            x,y = nx,ny
        else:
            dx,dy = -dy,dx
            x,y = x+dx, y+dy
    return myarray

def spiral_list_to_array(n):
    spiral_list = spiral(n)
    spiral_array = np.zeros(np.power(n,2))
    c = 0
    for y in range(len(spiral_list)):
        for x in range(len(spiral_list)):
            spiral_array[c] = int(spiral_list[x][y])
            c = c +1
    return spiral_array

def printspiral(myarray):
    n = range(len(myarray))
    for y in n:
        for x in n:
            print "%2i" % myarray[x][y],
        print
