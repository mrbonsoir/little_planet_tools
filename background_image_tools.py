# This module contains functions to create images with repeatitive patterns.
#   - pattern made of colored circles
#   - pattern made if colored stripes

import cv2
import numpy as np
import os
import shutil
import matplotlib.pyplot as plt
import halftoning_tools as ht

def create_circle_pattern(pattern_size, color_circles,
                          source_circle = "center",
                          number_circle = 8):
    """
    The function creates image made of colored circles where the color
    sequence is repated until the image is covered.

    In:
        pattern_size (numpy array): two element vectors giving the image pattern
        size
        color_circles (numpy array): size n x 3 for n different rgb color
        number_circle (int): a number defining how many circles will be plotted
        on the image pattern
    Out:
        circle_img (numpy array): image of size of the pattern_size but with 3
        channels for R, G and B.
    """
    width, height = pattern_size[0], pattern_size[1]

    if source_circle == "center":
        center = (width/2,height/2)

    elif source_circle == "random":
        center = (width * np.random.rand(1),height * np.random.rand(1))

    else:
        print "You choose to center all cirle on the %s location of the image" % source_circle

    # the longest distance is the diagonal of the picture
    r_max = np.round(np.sqrt(np.power(width,2) + np.power(height,2)))
    print r_max

    # computer circle step
    radius_step = width / (number_circle * 2)
    radius_vec = np.arange(0, int(r_max) * 1, radius_step)
    radius_vec = radius_vec[::-1]
    radius_vec = radius_vec[0:-1]
    circle_img = np.zeros((height, width,3), np.uint8)
    number_color = len(color_circles)

    # start to fill the image with colored circles.
    # first the large one, then over on it several other of smaller sizes.
    s = 0
    for r_step in radius_vec:
        #print r_step,
        color_circle = color_circles[s]
        cv2.circle(circle_img, center, int(r_step), color_circle, thickness=-1)
        s = s + 1

        if s == number_color:
            s = 0

    return circle_img

def create_stripes_pattern(pattern_size, color_stripes, number_stripes = 8, angle_stripe = 0,
    color_sun= (255, 255,255), circle_size = 0.1):
    """The function creates image made of colored stripes where the color
    sequence is repated until the image is covered.

    In:
        pattern_size (numpy array): two element vectors giving the image pattern
        size
        color_stripes (numpy array): size n x 3 for n different rgb color
        number_circle (int): a number defining how many circles will be plotted
        on the image pattern
        alpha (int): a parameter defining how the stripes are tilted

    Out:
        stripe_img (numpy array): image of size of the pattern_size but with 3
        channels for R, G and B.
    """

    number_color = len(color_stripes)

    # create a black image
    width, height = pattern_size[0], pattern_size[1]
    stripe_img = np.zeros((height, width,3), np.uint8)

    # parameter for the ing function
    contourIdx = -1

    # compute positions of the stripes and displacement according to angle value
    stripe_step = width / number_stripes
    stripe_vec1 = np.arange(-width, width*2, stripe_step)
    alpha = np.tan(angle_stripe * np.pi / 180) * height

    # start ing the stripes on the black picture
    s = 0
    for s_step in stripe_vec1:
        color_stripe = color_stripes[s]
        L = [[s_step, 0],                              # coordinates of the
             [alpha  + s_step, height],                # four corners figure
             [alpha  + s_step + stripe_step, height],  # corresponding to one
             [s_step + stripe_step, 0]]                # stripe.
        contours = np.array(L).reshape((-1,1,2)).astype(np.int32)
        cv2.drawContours(stripe_img,[contours],0,color_stripe,-1)
        s = s + 1
        if s == len(color_stripes):
            s = 0

    center_ = (width / 2,height / 2)
    r_max = np.sqrt((np.power(height,2)+np.power(width,2)))
    #color_sun = color_sun
    cv2.circle(stripe_img, center_, int(r_max * circle_size), color_sun, thickness=-1)

    return stripe_img

def create_circle_and_ray_of_light(pattern_size, color_stripes, angle_step = 60, angle_alpha = 0, circle_size = 0.1):
    """
    The function draws a pattern that looks like the sun in the center of the
    frame with manga style ray of light going in every directions from the
    center. Key word is obviously center.
    In:
        pattern_size (numpy array): two element vectors giving the image pattern
                                    size
        color_stripes (numpy array): size n x 3 for n different rgb color
        number_circle (int): a number defining how many circles will be plotted
                        on the image pattern
    Out:
        sunray_img (numpy array): image of size of the pattern_size but with 3
        channels for R, G and B.
    """

    width, height = pattern_size[0], pattern_size[1]
    center_ = (width / 2,height / 2)
    r_max = np.sqrt((np.power(height,2)+np.power(width,2)))
    color_sunry = color_stripes[1:]
    color_sun = color_stripes[0]

    # create black image
    sunray_img = np.zeros((height, width,3), np.uint8)

    vec_angle = np.hstack([np.arange(0 + angle_alpha, 360 + angle_alpha, angle_step), 360 + angle_alpha])
    print vec_angle
    s = 0
    for ii in np.arange(len(vec_angle)-1):
        angle_val = vec_angle[ii]
        angle_A = (vec_angle[ii] * np.pi ) / 180
        angle_B = (vec_angle[ii+1] * np.pi ) / 180
        L = [[width / 2, height / 2],
             [width / 2 + r_max * np.sin(angle_A), height / 2 + r_max * np.cos(angle_A)],
             [width / 2 + r_max * np.sin(angle_B), height / 2 + r_max * np.cos(angle_B)]]
        contours = np.array(L).reshape((-1,1,2)).astype(np.int32)
        cv2.drawContours(sunray_img,[contours],0,color_sunry[s],-1)
        s = s + 1
        if s == len(color_sunry):
            s = 0

    # draw sun in the center
    cv2.circle(sunray_img, center_, int(r_max * circle_size), color_sun, thickness=-1)

    return sunray_img

def display_image(image_data, image_name_for_title):
    """Thte function display an image, panorama image of little planet.
    """
    fig = plt.figure(figsize=(15,15))
    plt.imshow(image_data)
    plt.xticks([])
    plt.yticks([])
    plt.title(image_name_for_title)
    plt.show()

def apply_pattern_mask(image_data, pattern_image_data, equalize_parameter = False,
                        binary_inv = False, threshold_parameter = 128):
    """The function assumes a RGB images as input. It will threshold this image,
    then create a mask version of it and will combine it with colorfull pattern
    image.

    In:
        image_data (numpy array): n x m x3
        pattern_image_data (numpy array): n x m x 3
    Out:
        image_masked_and_pattern (numpy array): n x m x 3
    """

    # convert RGB to grayscale
    imG = cv2.cvtColor(image_data, cv2.COLOR_RGB2GRAY)

    # equalize or not
    if equalize_parameter == True:
        imG = cv2.equalizeHist(imG)

    # apply thresold to binarize the grayscale image
    if binary_inv == False:
        ret, th1 = cv2.threshold(imG,threshold_parameter,255,cv2.THRESH_BINARY)
    else:
        #th1 = ht.fun_halftone_image_with_mask(imG, mask_size=16, mask_type="linear")
        ret, th1 = cv2.threshold(imG,threshold_parameter,255,cv2.THRESH_BINARY_INV)

    # back to BGR for opencv
    pattern = cv2.cvtColor(pattern_image_data, cv2.COLOR_RGB2BGR)

    # apply mask
    th1 = 255 - th1
    masked_data = cv2.bitwise_and(pattern,pattern, mask=th1)
    #masked_data = cv2.bitwise_or(pattern,pattern, mask=th1)

    masked_data = cv2.cvtColor(masked_data, cv2.COLOR_BGR2RGB)
    return masked_data

def rotate_image(image_rgb, angle_rotation = 2):
    """The function rotate an image of angle_rotation degrees."""
    image_shape = np.shape(image_rgb)
    if len(image_shape) == 2:
        print "we have a BW image."
    else:
        print "we have a color image."
        #print image_shape
        image_rgb_rotated = image_rgb

    center = (image_shape[1] / 2, image_shape[0] / 2)
    angle_rotation = 2 # rotate the image by x degrees
    M1 = cv2.getRotationMatrix2D(center, angle_rotation, 1.0)
    image_rgb_rotated[:,:,1] = cv2.warpAffine(image_rgb[:,:,0], M1, (image_shape[1], image_shape[0]))
    M2 = cv2.getRotationMatrix2D(center, angle_rotation * 2, 1.0)
    image_rgb_rotated[:,:,2] = cv2.warpAffine(image_rgb[:,:,1], M2, (image_shape[1], image_shape[0]))

    return image_rgb_rotated



def fun_create_3x3_pixel_cell(pattern_size, color_patches, cell_factor, color_sun= (255, 255,255), circle_size = 0.1):
    """
    The function does things, one of them is to create a background image made of
    the repetition of a colorfull cell defined by its colors.

    It is assumed that the is of size 3 x 3.
    """

    # initiaize output
    width, height = pattern_size[0], pattern_size[1]
    img_pix = np.zeros((height, width,3), dtype="uint8")
    print np.shape(img_pix)

    #img_list = [cv2.imread(dir_images+fn) for fn in img_fn]
    channel_R = np.zeros((9,1),dtype="uint8")
    channel_G = np.zeros((9,1),dtype="uint8")
    channel_B = np.zeros((9,1),dtype="uint8")
    for ii in np.arange(9):
        channel_R[ii] = color_patches[ii][0]
        channel_G[ii] = color_patches[ii][1]
        channel_B[ii] = color_patches[ii][2]

    # reshape in cell of 3x3
    channel_R = np.reshape(channel_R,(3,3))
    channel_G = np.reshape(channel_G,(3,3))
    channel_B = np.reshape(channel_B,(3,3))

    # increase the size of the cell
    mask_size = pattern_size[0] / cell_factor
    channel_R = cv2.resize(channel_R, (mask_size, mask_size), interpolation = cv2.INTER_NEAREST)
    channel_G = cv2.resize(channel_G, (mask_size, mask_size), interpolation = cv2.INTER_NEAREST)
    channel_B = cv2.resize(channel_B, (mask_size, mask_size), interpolation = cv2.INTER_NEAREST)

    # expand to final size
    factor_width  = int(np.ceil(np.shape(img_pix)[0] / mask_size)+1)
    factor_height = int(np.ceil(np.shape(img_pix)[1] / mask_size)+1)

    print factor_width, factor_height
    channel_R = np.tile(channel_R,(factor_width, factor_height))
    channel_G = np.tile(channel_G,(factor_width, factor_height))
    channel_B = np.tile(channel_B,(factor_width, factor_height))
    print np.shape(channel_B)

    # create final image
    img_pix[:,:,0] = channel_R[0:np.shape(img_pix)[0],0:np.shape(img_pix)[1]]
    img_pix[:,:,1] = channel_G[0:np.shape(img_pix)[0],0:np.shape(img_pix)[1]]
    img_pix[:,:,2] = channel_B[0:np.shape(img_pix)[0],0:np.shape(img_pix)[1]]
    print np.shape(img_pix)

    center_ = (width / 2,height / 2)
    r_max = np.sqrt((np.power(height,2)+np.power(width,2)))
    color_sun = color_sun
    cv2.circle(img_pix, center_, int(r_max * circle_size), color_sun, thickness=-1)

    return img_pix

def fun_combine_pix_square(im1, im2, im3):
    """The function combines different pixel version of the little planet image.
    It put in the center the highest resolution and lower to the surrounding. """
    imOut = im3
    imOut[:, 1500:7500,:] = im2[:, 1500:7500,:]
    imOut[1500:4500, 3000:6000,:] = im1[1500:4500, 3000:6000,:]
    return imOut

def fun_combine_pix_stripes(im1,im2,im3):
    """The function combines the pixel version of the little planet into stripes. Run it
    and you will see."""
    imOut = im3
    imOut[:, 1500:7500,:] = im2[:, 1500:7500,:]
    imOut[:, 3000:6000,:] = im1[:, 3000:6000,:]

    return imOut


def fun_combine_pix(im1, im2, im3):
    """The function combines the pixel version of the litllet planet.
    The function assumes all images im1, im2 and im3 having the same size
    and being of size 6000x9000x3."""
    imOut = np.zeros(np.shape(im1),dtype=imCrop5.dtype)
    imOut = im2
    #center
    imOut[1500:4500, 3000:6000,:] = im1[1500:4500, 3000:6000,:]
    # corner
    imOut[0:1500,0:3000,:] = im3[0:1500,0:3000:,:]
    imOut[0:1500,-3000:,:] = im3[0:1500,-3000:,:]
    imOut[-1500:,0:3000,:] = im3[-1500:,0:3000,:]
    imOut[-1500:,-3000:,:] = im3[-1500:,-3000:,:]

    return imOut

def fun_combined_2images_with_circle_mask(list_image, radius):
    """the function does some masking in a smart way.
    """

    # create list of mask
    imMask = np.zeros(np.shape(list_image[0])[0:2], dtype=list_image[0].dtype)

    # add the circle
    width, height = np.shape(imMask)[1], np.shape(imMask)[0]
    center_ = (width / 2, height / 2)
    color_ = 255
    if radius < 1:
        radius_ = int(width / 2 * radius)
    else:
        radius_ = radius
    print "radius is %1.0f." % radius_
    cv2.circle(imMask, center_, radius_, color_, thickness=-1)

    # loop ove the images
    masked_data = cv2.bitwise_and(list_image[0], list_image[0], mask=imMask)
    imMask_inv = cv2.bitwise_not(imMask)
    masked_data_inv = cv2.bitwise_and(list_image[1], list_image[1], mask=imMask_inv)

    im_combined = masked_data + masked_data_inv

    return im_combined

def fun_create_tile_littleplanet_wallpaper(img_lp, factor_tile = 1):
    """The function create an image made of the repition of a square tile.
    The square tile is black and white or color version of the little planet
    image **img_lp**.
    The output image can easily be used as mask image for later processing.
    IN:
        factor_tile (int): a factor that defines the size of the tile assuming
        a given image of siez 6000x9000.
    OUT:
        img_lp_tile (mxnx3 uint8): an image of the same size as the inpute img_lp
    """

    # select square part of the input image
    square_RGB = img_lp[:,1500:7500,:]

    # resize square to the tile size
    size_tile = (3000/ factor_tile, 3000 / factor_tile)
    tile_RGB = cv2.resize(square_RGB, size_tile, interpolation = cv2.INTER_CUBIC)

    # create uniform color image
    tile_background = np.zeros(np.shape(tile_RGB), dtype="uint8")
    tile_background[:,:,0] = 255
    tile_background[:,:,1] = 255
    tile_background[:,:,2] = 255

    # one single tile
    combined_tile_RGB = apply_pattern_mask(tile_RGB, tile_background,
                                            equalize_parameter = False,
                                            binary_inv = True,
                                            threshold_parameter = 128)
    #print np.shape(combined_tile_RGB)

    # check some potential sise issue
    resize_tile = (2 * factor_tile, 3* factor_tile)
    tmp = np.tile(combined_tile_RGB[:,:,0],resize_tile)

    if np.shape(tmp)[0]-np.shape(img_lp)[0] != 0 and np.shape(tmp)[1]-np.shape(img_lp)[1] != 0:
        img_tile_RGB = np.zeros((np.shape(tmp)[0],np.shape(tmp)[1],3), dtype="uint8")
        for ii in np.arange(3):
            img_tile_RGB[:,:,ii] = np.tile(combined_tile_RGB[:,:,ii],resize_tile)

        # resize to image original size
        img_tile_RGB = cv2.resize(img_tile_RGB,
                                  (np.shape(img_lp)[1],np.shape(img_lp)[0]),
                                  interpolation = cv2.INTER_CUBIC)

    else: # no siez problem
        img_tile_RGB = np.zeros(np.shape(img_lp), dtype="uint8")
        for ii in np.arange(3):
            img_tile_RGB[:,:,ii] = np.tile(combined_tile_RGB[:,:,ii],resize_tile)

    return img_tile_RGB



def fun_create_tile_littleplanet_wallpaper_with_shift(img_lp, factor_tile = 1):
    """The function create an image made of the repetition of a square tile but
    in addition we have a shit one every two lines.
    The square tile is black and white or color version of the little planet
    image **img_lp**.
    The output image can easily be used as mask image for later processing.
    IN:
        factor_tile (int): a factor that defines the size of the tile assuming
        a given image of siez 6000x9000.
    OUT:
        img_lp_tile (mxnx3 uint8): an image of the same size as the inpute img_lp
    """

    # select square part of the input image
    square_RGB = img_lp[:,1500:7500,:]

    # resize square to the tile size
    size_tile = (3000/ factor_tile, 3000 / factor_tile)
    tile_RGB = cv2.resize(square_RGB, size_tile, interpolation = cv2.INTER_CUBIC)

    # create uniform color image
    tile_background = np.zeros(np.shape(tile_RGB), dtype="uint8")
    tile_background[:,:,0] = 255
    tile_background[:,:,1] = 255
    tile_background[:,:,2] = 255

    combined_tile_RGB = apply_pattern_mask(tile_RGB, tile_background,
                                            equalize_parameter = False,
                                            binary_inv = True,
                                            threshold_parameter = 128)

    size_tile = np.shape(combined_tile_RGB)
    combined_tile_RGB_shift = np.hstack([combined_tile_RGB[:,int(size_tile[1]/2):,:],
                                        combined_tile_RGB[:,0:int(size_tile[1]/2),:],])

    resize_tile = (2 * factor_tile, 3* factor_tile)
    img_tile_RGB = np.zeros(np.shape(img_lp), dtype="uint8")
    vec = np.hstack([np.arange(0,np.shape(img_lp)[0],size_tile[0]),np.shape(img_lp)[0]])

    # check size issue
    tmp = np.tile(combined_tile_RGB[:,:,0],resize_tile)
    if np.shape(tmp)[0]-np.shape(img_lp)[0] != 0 and np.shape(tmp)[1]-np.shape(img_lp)[1] != 0:
        # create one line of normal and shifted tile
        img_tile_RGB = np.zeros((np.shape(tmp)[0],np.shape(tmp)[1],3), dtype="uint8")

        k = 0
        for jj in np.arange(resize_tile[0]):
            if jj % 2 == 0:
                for ii in np.arange(3):
                    img_tile_RGB[vec[k]:vec[k+1],:,ii] = \
                            np.tile(combined_tile_RGB[:,:,ii],(1,resize_tile[1]))
            else:
                for ii in np.arange(3):
                    img_tile_RGB[vec[k]:vec[k+1],:,ii] = \
                        np.tile(combined_tile_RGB_shift[:,:,ii],(1,resize_tile[1]))
            k = k + 1
        img_tile_RGB = cv2.resize(img_tile_RGB,
                                  (np.shape(img_lp)[1],np.shape(img_lp)[0]),
                                  interpolation = cv2.INTER_CUBIC)

    else: #no size problem
        k = 0
        band_tile_RGB = np.zeros((np.shape(tile_RGB)[0],np.shape(tmp)[1],3), dtype="uint8")
        band_tile_RGB_shift = np.zeros((np.shape(tile_RGB)[0],np.shape(tmp)[1],3), dtype="uint8")

        for jj in np.arange(resize_tile[0]):
            if jj % 2 == 0:
                for ii in np.arange(3):
                    img_tile_RGB[vec[k]:vec[k+1],:,ii] = \
                            np.tile(combined_tile_RGB[:,:,ii],(1,resize_tile[1]))
            else:
                for ii in np.arange(3):
                    img_tile_RGB[vec[k]:vec[k+1],:,ii] = \
                        np.tile(combined_tile_RGB_shift[:,:,ii],(1,resize_tile[1]))
            k = k + 1

    return img_tile_RGB


def fun_change_single_image_color(img_RGB, old_color = (0,0,0), new_color=(128,128,128)):
    """As it stands all pixel with the old_color value will be change to the new_color
    value. For now the function works best with BW image given as input where the black
    pixels are replaced the the new_color value.
    IN:
        old_color (int, int, int): RGB color value [0,255]
        new_color (int, int, int): RGB color value [0,255]
     OUT:
         img_RGB2 modidief image.
     """

    tmp_ = np.zeros((np.shape(img_RGB)[0]*np.shape(img_RGB)[1],3),dtype=img_RGB.dtype)
    for ii in np.arange(3):
        tmp_[:,ii] = img_RGB[:,:,ii].flatten()

    #ind_ = np.where(np.sum(tmp_,axis=1) == 0)[0]
    ind_ = np.where(np.sqrt(np.power(tmp_[:,0] - old_color[0],2) +
                            np.power(tmp_[:,1] - old_color[1],2) +
                            np.power(tmp_[:,2] - old_color[0],2)) == 0)

    # change color
    print new_color
    for ii in np.arange(3):
        tmp_[ind_,ii] = new_color[ii]

    # resize the bazard
    img_RGB2 = np.zeros(np.shape(img_RGB), dtype=img_RGB.dtype)
    for ii in np.arange(3):
        img_RGB2[:,:,ii] = np.reshape(tmp_[:,ii],(np.shape(img_RGB)[0],np.shape(img_RGB)[1]))

    return img_RGB2

def fun_change_single_image_color_by_other_image(img_RGB, img_background, old_color = (0,0,0)):
    """As it stands all pixel with the old_color value will be change to the new_color
    value. For now the function works best with BW image given as input where the black
    pixels are replaced the the new_color value.
    IN:
        old_color (int, int, int): RGB color value [0,255]
        new_color (int, int, int): RGB color value [0,255]
     OUT:
         img_RGB2 modidief image.
     """

    tmp_ = np.zeros((np.shape(img_RGB)[0]*np.shape(img_RGB)[1],6),dtype=img_RGB.dtype)
    for ii in np.arange(3):
        tmp_[:,ii] = img_RGB[:,:,ii].flatten()
        tmp_[:,ii+3] = img_background[:,:,ii].flatten()

    #ind_ = np.where(np.sum(tmp_,axis=1) == 0)[0]
    ind_ = np.where(np.sqrt(np.power(tmp_[:,0] - old_color[0],2) +
                            np.power(tmp_[:,1] - old_color[1],2) +
                            np.power(tmp_[:,2] - old_color[0],2)) == 0)

    # change color
    tmp_[ind_,0:2] = tmp_[ind_,3:6]

    # resize the bazard
    img_RGB2 = np.zeros(np.shape(img_RGB), dtype=img_RGB.dtype)
    for ii in np.arange(3):
        img_RGB2[:,:,ii] = np.reshape(tmp_[:,ii],(np.shape(img_RGB)[0],np.shape(img_RGB)[1]))

    return img_RGB2
