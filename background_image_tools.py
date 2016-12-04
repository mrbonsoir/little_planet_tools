# This module contains functions to create images with repeatitive patterns.
#   - pattern made of colored circles
#   - pattern made if colored stripes

import cv2
import numpy as np
import os
import shutil
import matplotlib.pyplot as plt

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

def create_stripes_pattern(pattern_size, color_stripes, number_stripes = 8, angle_stripe = 0):
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

    # create black image
    sunray_img = np.zeros((height, width,3), np.uint8)

    vec_angle = np.hstack([np.arange(0 + angle_alpha, 360 + angle_alpha, angle_step), 360 + angle_alpha])

    s = 0
    for ii in np.arange(len(vec_angle)-1):
        angle_val = vec_angle[ii]
        angle_A = (vec_angle[ii] * np.pi ) / 180
        angle_B = (vec_angle[ii+1] * np.pi ) / 180
        L = [[width / 2, height / 2],
             [width / 2 + r_max * np.sin(angle_A), height / 2 + r_max * np.cos(angle_A)],
             [width / 2 + r_max * np.sin(angle_B), height / 2 + r_max * np.cos(angle_B)]]
        contours = np.array(L).reshape((-1,1,2)).astype(np.int32)
        cv2.drawContours(sunray_img,[contours],0,color_stripes[s],-1)
        s = s + 1
        if s == len(color_stripes):
            s = 0

    # draw sun in the center
    cv2.circle(sunray_img, center_, int(r_max * circle_size), color_stripes[0], thickness=-1)

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
