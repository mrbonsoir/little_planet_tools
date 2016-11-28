import cv2
import numpy as np
import os
import shutil
import matplotlib.pyplot as plt
import skimage
from skimage import exposure

dir_tmp = "/Users/jeremie/Pictures/tmp/"

class LittlePlanet:
    """This class defines a LittlePlanet object"""

    """. When a full spherical
    panorama image is given as an equirectangular image, then a configuration
    file is automatically created and used to generate a first version of
    little planet image.

    Attibutes:
        distance: it defined how big/small it appears
        angle: it defines how much is the final LittlePlanet rotated around the
        z axis
    """

    def __init__(self, path_and_name_equirectangular, distance=300, angle=90,
                 pano_output_width = 1000, pano_output_height = 500,
                 file_name_suffix = "thumb"):
        """Return a a LittlePlanet object whose distance is *distance* and
        and angle is *angle*."""

        # check if the image has a ratio of 2 and if not print a warning
        file_path, file_name = os.path.split(path_and_name_equirectangular)

        self.pano_equirectangular_name = file_name
        self.file_name_suffix = file_name_suffix
        print "Panorama equirectangular name: %s" % self.pano_equirectangular_name

        # load the image
        self.im_pano_equirectangular = cv2.imread(path_and_name_equirectangular)
        self.im_pano_equirectangular = cv2.cvtColor(self.im_pano_equirectangular,
                                                    cv2.COLOR_BGR2RGB)

        img_size = np.shape(self.im_pano_equirectangular)
        ratio_width_height = img_size[1] / img_size[0]
        if ratio_width_height == 2:
            self.pano_width = img_size[1]
            self.pano_height = img_size[0]
            print "Panorama equirectangular width: %1.0f " %  self.pano_width
            print "Panorama equirectangular height: %1.0f " % self.pano_height
            print "Suffix panorama name: %s " % self.file_name_suffix

        else:
            print "You should do something with your life."

        # resize image to get a faster manageable version
        self.pano_output_width = pano_output_width
        self.pano_output_height = pano_output_height
        self.im_pano_equirectangular = cv2.resize(self.im_pano_equirectangular,
                                                  (self.pano_output_width,
                                                  self.pano_output_height),
                                                  interpolation = cv2.INTER_CUBIC)
        # save thumb as a jpg image somewhere
        self.im_pano_equirectangular = cv2.cvtColor(self.im_pano_equirectangular,
                                                    cv2.COLOR_BGR2RGB)
        cv2.imwrite(dir_tmp+"pano_equi_"+self.file_name_suffix+".jpg", self.im_pano_equirectangular)

        # basic parameters for the config file
        self.distance = distance
        self.r_angle = -90
        self.p_angle = angle
        self.y_angle = -90
        self.var_output_width = self.pano_output_width
        self.var_output_height = self.pano_output_height
        self.output_file_name = "little_planet_"+self.file_name_suffix+".jpg"

        # print some info about the little planet
        print "Little image in progress with following parameters:"
        print "output_width: %1.0f" % self.pano_output_width
        print "output_height %1.0f" % self.pano_output_height
        print "output_distance: %1.0f" % self.distance
        print "r_angle: %1.0f" % self.r_angle
        print "p_angle: %1.0f" % self.p_angle
        print "y_angle: %1.0f" % self.y_angle
        print "output name: %s" % self.output_file_name

        # with _init__ a config file should be automatically created
        create_pto_config_file(path_and_name_equirectangular,
                            output_file_name = "template333",
                            output_width = self.pano_output_width,
                            output_height = self.pano_output_height,
                            r_angle = self.r_angle,
                            p_angle = self.p_angle,
                            y_angle = self.y_angle,
                            output_distance = self.distance)

        # save as above we generate directly a little planet
        create_little_planet(path_and_name_equirectangular,
                            config_file_pto = "template333.pto",
                            little_planet_name = self.output_file_name)

    def do_color_work(self, little_planet_name):
        """The function applies some color transformation on the resulting little planet
        image that has been created."""

        # convert to BW to bin
        im_little_planet = cv2.imread(little_planet_name)
        im_little_planet_BW = cv2.cvtColor(im_little_planet, cv2.COLOR_BGR2GRAY)
        cv2.imwrite(little_planet_name[0:-4]+"_BW.jpg", im_little_planet_BW)
        self.im_little_planet_BW = im_little_planet_BW

        im_little_planet_BW_bin = cv2.GaussianBlur(im_little_planet_BW,(5,5),0)
        ret,th1 = cv2.threshold(im_little_planet_BW_bin,127,255,cv2.THRESH_BINARY)
        cv2.imwrite(little_planet_name[0:-4]+"_BW_bin.jpg",th1)

        # convert BW to BW equalize to bin
        im_little_planet_BW_equalize = cv2.equalizeHist(im_little_planet_BW)
        cv2.imwrite(little_planet_name[0:-4]+"_BW_equalize.jpg",im_little_planet_BW_equalize)

        # Convert BW equalize to binary Otsu's thresholding after Gaussian filtering
        im_little_planet_BW_equalize_bin = cv2.GaussianBlur(im_little_planet_BW_equalize,(5,5),0)
        ret,th11 = cv2.threshold(im_little_planet_BW_equalize_bin,127,255,cv2.THRESH_BINARY)
        cv2.imwrite(little_planet_name[0:-4]+"_BW_equalize_bin.jpg",th11)


        # do some morphological operation on the binaray image
        kernel = np.ones((5,5),np.uint8)
        opening_bin = cv2.morphologyEx(th1, cv2.MORPH_OPEN, kernel)
        cv2.imwrite(little_planet_name[0:-4]+"_BW_bin_opening.jpg",opening_bin)

        opening_equalize_bin = cv2.morphologyEx(th11, cv2.MORPH_OPEN, kernel)
        cv2.imwrite(little_planet_name[0:-4]+"_BW_equalize_bin_opening.jpg",opening_equalize_bin)

        # rotation of n degrees the color and BW images
        im_little_planet_rotation = rotate_image(im_little_planet)
        cv2.imwrite(little_planet_name[0:-4]+"_rotation.jpg",im_little_planet_rotation)

    def do_pattern_color_work(self, little_planet_name,
                                color_circles,
                                color_stripes,
                                alpha_stripe = 0,
                                number_circle = np.array([8, 16, 32]),
                                number_stripes = np.array([8, 16, 32])):
        """The function will create test pattern and apply them to the little
        planet chosen by its name little_planet_name.

        The pattern image will be saved in the same folder as the little planets.
        """
        #print "ca vient ---> %1.f x %1.0f" % (self.pano_output_width, self.pano_output_height)
        # create the pattern and apply the pattern
        # circle 2 colors

        for i, nb_cir_str in enumerate(zip(number_circle, number_stripes)):
            #print i, nb_cir_str[0], nb_cir_str[1]
            # circle
            create_circle_pattern2((self.pano_output_width, self.pano_output_height),
                            color_circles[0:2], number_circle =  nb_cir_str[0])
            masked_data  = apply_pattern_mask(little_planet_name, "pattern_circle.jpg")
            cv2.imwrite(dir_tmp+"little_thumb_pattern_circle_"+str(i+1)+".jpg",masked_data)

            create_circle_pattern2((self.pano_output_width, self.pano_output_height),
                            color_circles, number_circle =  nb_cir_str[0])
            masked_data  = apply_pattern_mask(little_planet_name, "pattern_circle.jpg")
            cv2.imwrite(dir_tmp+"little_thumb_pattern_circle_"+str(i+4)+".jpg",masked_data)

            # stripes
            create_stripes_pattern2((self.pano_output_width, self.pano_output_height),
                                    color_stripes[0:2],
                                    number_stripes = nb_cir_str[1],
                                    alpha = alpha_stripe)
            masked_data  = apply_pattern_mask(little_planet_name, "pattern_stripes.jpg")
            cv2.imwrite(dir_tmp+"little_thumb_pattern_stripes_"+str(i+1)+".jpg",masked_data)

            create_stripes_pattern2((self.pano_output_width, self.pano_output_height),
                                    color_stripes,
                                    number_stripes = nb_cir_str[1],
                                    alpha = alpha_stripe)
            masked_data  = apply_pattern_mask(little_planet_name, "pattern_stripes.jpg")
            cv2.imwrite(dir_tmp+"little_thumb_pattern_stripes_"+str(i+4)+".jpg",masked_data)


# A few functions not specially linked to the class
def display_image(image_data, image_name_for_title):
    """Thte function display an image, panorama image of little planet.
    """
    fig = plt.figure(figsize=(15,15))
    plt.imshow(image_data)
    plt.xticks([])
    plt.yticks([])
    plt.title(image_name_for_title)
    plt.draw()
    plt.show()

def create_pto_config_file(path_and_name_equirectangular, output_file_name = "template33",
                           output_width = 3000, output_height = 1500,
                           r_angle = 0, p_angle = 90, y_angle = 0,
                           output_distance = 300):
    """This function create a pto file for hugin that can be used with nona.

    It uses a template in which we can change several parameters.
    What doesn't change is that we start from an equirectangular panorama that we remap to
    a spherical stereographic or something like that. Point is we are in the middle of a sphere
    and we want to apply some distorsion to make it appear like a litte planet.

    input:
        pano_equirectangular_name (char) : name of the panorama image you want to create little planet from.
        output_fileName (char) : "template33" | name of the pto filem can be changed
        image_width   (float) : 3000  | size for the input image (not sure it is necessary)
        image_height  (float) : 1500  |
        output_width  (float) : 3000  | size for the output little planet image
        output_height (float) : 1500  |
        r_angle (float) : 0           | angle values to make the initial spherical panorama to rotate before projection on plan
        p_angle (float) : 90          |
        y_angle (float) : 0           |
        output_distance (float) : 300 | tell how small(360)/big(0) will appear the little planet

    About r_angle, p_angle and y_angle, they are key parameters to control the rotationo of the little planet:
        - r_angle =   0, p_angle =  90, y_angle =   0 (default)
        - r_angle =   0, p_angle = -90, y_angle =   0 for an inverse little planet.
        - r_angle = -90, p_angle =   X, y_angle = -90 for a rotation clocklwise of X degrees.
        - r_angle =  90, p_angle =   X, y_angle = -90 for a rotation clockwise of an inverse little planet.
    """

    # get the size parameters from the pano equirectangular
    im_pano_equirectangular = cv2.imread(path_and_name_equirectangular)
    img_size = np.shape(im_pano_equirectangular)
    #print img_size, np.shape(img_size)
    image_width  = img_size[1]
    image_height = img_size[0]

    #hugin_ptoversion 2
    f = open(output_file_name+'.pto', 'w')
    f.write('# hugin project file\n')
    f.write('p f4 w'+str(output_width)+' h'+str(output_height)+' v'+str(output_distance)+' E0 R0 n"TIFF_m c:LZW r:CROP"\n')
    f.write('m g1 i0 f0 m2 p0.00784314\n')

    # image lines
    #-hugin  cropFactor=1
    f.write('i w'+str(image_width)+' h'+str(image_height)+' f4 v360 Ra0 Rb0 Rc0 Rd0 Re0 Eev0 Er1 Eb1 r'+str(r_angle)+' p'+str(p_angle)+' y'+str(y_angle)+' TrX0 TrY0 TrZ0 Tpy0 Tpp0 j0 a0 b0 c0 d0 e0 g0 t0 Va1 Vb0 Vc0 Vd0 Vx0 Vy0  Vm5 n"'+path_and_name_equirectangular+'"\n')

    # specify variables that should be optimized
    f.write('v Ra0\n')
    f.write('v Rb0\n')
    f.write('v Rc0\n')
    f.write('v Rd0\n')
    f.write('v Re0\n')
    f.write('v Vb0\n')
    f.write('v Vc0\n')
    f.write('v Vd0\n')
    f.write('v\n')

    f.close()

def create_little_planet(path_and_name_equirectangular, config_file_pto = "template33.pto",
                         little_planet_name = "little_planet_thumb"):
    """The function will call the funcion create_pto_config_file to be able to
    create the final little planet."""

    # apply command
    command_line_for_Hugin = "/Applications/Hugin/HuginTools/nona -o \
    /Users/jeremie/Pictures/tmp/imLittlePlanet33 -m TIFF %s %s " \
    % (config_file_pto,path_and_name_equirectangular)
    print "Command line to call Hugin:"
    print ">%s" % command_line_for_Hugin
    os.system(command_line_for_Hugin)

    # convert the TIFF file to jpg
    im_little_planet_tiff = cv2.imread("/Users/jeremie/Pictures/tmp/imLittlePlanet33.tif")
    cv2.imwrite("/Users/jeremie/Pictures/tmp/"+little_planet_name,im_little_planet_tiff)
    # remove the TIFF image
    os.remove("/Users/jeremie/Pictures/tmp/imLittlePlanet33.tif")

def create_circle_pattern(pattern_size, color1 , color2, number_circle = 8):
    """The function creates a bi-color circle parttern from the image center."""
    width, height = pattern_size[0], pattern_size[1]
    #print "size pattern %1.f x %1.0f" % (width, height)
    #print color1
    #print color2

    center = (width/2,height/2)
    radius_step = width / number_circle
    radius_vec = np.arange(width, 0 , -radius_step)
    circle_img = np.zeros((height, width,3), np.uint8)

    for i, r_step in enumerate(radius_vec):
        #print i, r_step
        if i % 2 == 0:
            cv2.circle(circle_img, center, r_step, color1, thickness=-1)
        else:
            cv2.circle(circle_img, center, r_step, color2, thickness=-1)

    circle_img = cv2.cvtColor(circle_img, cv2.COLOR_BGR2RGB)
    cv2.imwrite(dir_tmp+"pattern_circle.jpg",circle_img)
    return circle_img

def create_circle_pattern_3_colors(pattern_size, color1 , color2, color3, number_circle = 8):
    """The function creates a bi-color circle parttern from the image center."""
    width, height = pattern_size[0], pattern_size[1]
    #print "size pattern %1.f x %1.0f" % (width, height)
    #print color1
    #print color2

    center = (width/2,height/2)
    radius_step = width / number_circle
    radius_vec = np.arange(width, 0 , -radius_step)
    circle_img = np.zeros((height, width,3), np.uint8)

    c = 0
    for i, r_step in enumerate(radius_vec):
        #print i, r_step
        if c == 0:
            cv2.circle(circle_img, center, r_step, color1, thickness=-1)
            c = c + 1
        elif c == 1:
            cv2.circle(circle_img, center, r_step, color2, thickness=-1)
            c = c +1
        elif c == 2:
            cv2.circle(circle_img, center, r_step, color3, thickness=-1)
            c = 0

    circle_img = cv2.cvtColor(circle_img, cv2.COLOR_BGR2RGB)
    cv2.imwrite(dir_tmp+"pattern_circle.jpg",circle_img)
    return circle_img

def create_circle_pattern2(pattern_size, color_circles, number_circle = 8):
    """The function creates a bi-color circle parttern from the image center."""
    width, height = pattern_size[0], pattern_size[1]

    center = (width/2,height/2)
    radius_step = width / number_circle
    radius_vec = np.arange(width, radius_step , -radius_step)
    circle_img = np.zeros((height, width,3), np.uint8)

    number_color = len(color_circles)
    print number_color
    s = 0
    for r_step in radius_vec:
        color_circle = color_circles[s]
        cv2.circle(circle_img, center, r_step, color_circle, thickness=-1)
        s = s + 1

        if s == number_color:
            s = 0

    circle_img = cv2.cvtColor(circle_img, cv2.COLOR_BGR2RGB)
    cv2.imwrite(dir_tmp+"pattern_circle.jpg",circle_img)
    return circle_img


def create_stripes_pattern(pattern_size, color1, color2, number_stripes = 8, alpha = 0):
    """The function creates a bi-color circle parttern from the image center.
    """
    #print color1
    #print color2

    width, height = pattern_size[0], pattern_size[1]
    #print "size pattern %1.f x %1.0f" % (width, height)
    #center = (width/2,height/2)
    #radius_step = width / number_stripes
    #radius_vec = np.arange(width, 0 , -radius_step)
    stripes_img = np.zeros((height, width,3), np.uint8)

    contourIdx = -1

    stripe_step = width / number_stripes

    stripe_vec1 = np.arange(-width, width, stripe_step*2)
    stripe_vec2 = np.arange(stripe_step-width, width, stripe_step*2)

    #print alpha
    for s_step in stripe_vec1:
        L = [[s_step, 0],
             [alpha + s_step, height],
             [alpha + s_step+stripe_step, height],
             [s_step+stripe_step, 0]]
        contours = np.array(L).reshape((-1,1,2)).astype(np.int32)
        cv2.drawContours(stripes_img,[contours],0,color1,-1)

    for s_step in stripe_vec2:
        L = [[s_step, 0],
             [alpha + s_step, height],
             [alpha + s_step + stripe_step, height],
             [s_step + stripe_step, 0]]
        contours = np.array(L).reshape((-1,1,2)).astype(np.int32)
        cv2.drawContours(stripes_img,[contours],0,color2,-1)

    stripes_img = cv2.cvtColor(stripes_img, cv2.COLOR_BGR2RGB)
    cv2.imwrite(dir_tmp+"pattern_stripes.jpg", stripes_img)
    return stripes_img

def create_stripes_pattern2(pattern_size, color_stripes, number_stripes = 8, alpha = 0):
    """The function creates a bi-color circle parttern from the image center.
    """
    #print color1
    #print color2

    number_color = len(color_stripes)

    width, height = pattern_size[0], pattern_size[1]
    stripes_img = np.zeros((height, width,3), np.uint8)

    contourIdx = -1

    stripe_step = width / number_stripes

    stripe_vec1 = np.arange(-width, width, stripe_step)
    stripe_vec2 = np.arange(stripe_step-width, width, stripe_step)

    #print alpha
    s = 0
    for s_step in stripe_vec1:

        color_stripe = color_stripes[s]
        L = [[s_step, 0],
             [alpha  + s_step, height],
             [alpha  + s_step + stripe_step, height],
             [s_step + stripe_step, 0]]
        contours = np.array(L).reshape((-1,1,2)).astype(np.int32)
        cv2.drawContours(stripes_img,[contours],0,color_stripe,-1)
        s = s + 1
        if s == len(color_stripes):
            s = 0

    stripes_img = cv2.cvtColor(stripes_img, cv2.COLOR_BGR2RGB)
    cv2.imwrite(dir_tmp+"pattern_stripes.jpg", stripes_img)
    return stripes_img


def create_stripes_pattern_orthogonal(pattern_size, color1, color2, number_stripes = 8, alpha = 0):
    """The function creates a bi-color circle parttern from the image center.
    """
    #print color1
    #print color2

    width, height = pattern_size[0], pattern_size[1]
    #print "size pattern %1.f x %1.0f" % (width, height)
    #center = (width/2,height/2)
    #radius_step = width / number_stripes
    #radius_vec = np.arange(width, 0 , -radius_step)
    stripes_img = np.zeros((height, width,3), np.uint8)

    contourIdx = -1

    stripe_step = width / number_stripes
    stripe_vec1 = np.arange(-width, width, stripe_step*2)
    stripe_vec2 = np.arange(stripe_step-width, width, stripe_step*2)
    #print alpha
    for s_step in stripe_vec1:
        L = [[s_step, 0],
             [alpha + s_step, height],
             [alpha + s_step+stripe_step, height],
             [s_step+stripe_step, 0]]
        contours = np.array(L).reshape((-1,1,2)).astype(np.int32)
        cv2.drawContours(stripes_img,[contours],0,color1,-1)

    for s_step in stripe_vec2:
        L = [[s_step, 0],
             [alpha + s_step, height],
             [alpha + s_step + stripe_step, height],
             [s_step + stripe_step, 0]]
        contours = np.array(L).reshape((-1,1,2)).astype(np.int32)
        cv2.drawContours(stripes_img,[contours],0,color2,-1)

    stripes_img = cv2.cvtColor(stripes_img, cv2.COLOR_BGR2RGB)
    cv2.imwrite(dir_tmp+"pattern_stripes.jpg", stripes_img)
    return stripes_img



def create_stripes_pattern_3_colors(pattern_size, color1, color2, color3, number_stripes = 8, alpha = 0):
    """The function creates a tri-color circle parttern from the image center.
    """

    width, height = pattern_size[0], pattern_size[1]
    stripes_img = np.zeros((height, width,3), np.uint8)

    contourIdx = -1

    stripe_step = width / number_stripes
    stripe_vec1 = np.arange(-width, width, stripe_step*3)
    stripe_vec2 = np.arange(stripe_step-width, width, stripe_step*3)
    stripe_vec3 = np.arange(2*stripe_step-width, width, stripe_step*3)

    #print "The alpha parameter is %1.0f" % alpha
    for s_step in stripe_vec1:
        L = [[s_step, 0],
             [alpha + s_step, height],
             [alpha + s_step+stripe_step, height],
             [s_step+stripe_step, 0]]
        contours = np.array(L).reshape((-1,1,2)).astype(np.int32)
        cv2.drawContours(stripes_img,[contours],0,color1,-1)

    for s_step in stripe_vec2:
        L = [[s_step, 0],
             [alpha + s_step, height],
             [alpha + s_step + stripe_step, height],
             [s_step + stripe_step, 0]]
        contours = np.array(L).reshape((-1,1,2)).astype(np.int32)
        cv2.drawContours(stripes_img,[contours],0,color2,-1)

    for s_step in stripe_vec3:
        L = [[s_step, 0],
             [alpha + s_step, height],
             [alpha + s_step + stripe_step, height],
             [s_step + stripe_step, 0]]
        contours = np.array(L).reshape((-1,1,2)).astype(np.int32)
        cv2.drawContours(stripes_img,[contours],0,color3,-1)

    stripes_img = cv2.cvtColor(stripes_img, cv2.COLOR_BGR2RGB)
    cv2.imwrite(dir_tmp+"pattern_stripes.jpg", stripes_img)
    return stripes_img

def apply_pattern_mask(little_planet_filename, pattern_filename):
    """The function apply a colorfull patternish mask to the binaray
    version of the little planet."""

    #print " -->%s" %  little_planet_filename
    im = cv2.imread(little_planet_filename)
    imG = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    ret, th1 = cv2.threshold(imG,127,255,cv2.THRESH_BINARY)
    #th1_inv = cv2.bitwise_not(th1)

    height,width,depth = im.shape
    #print "--> %s  --" % (dir_tmp+pattern_filename)
    pattern = cv2.imread(dir_tmp+pattern_filename)
    #print np.shape(pattern), np.shape(th1)
    pattern = cv2.cvtColor(pattern, cv2.COLOR_RGB2BGR)

    # apply mask
    th1 = 255 - th1
    masked_data = cv2.bitwise_and(pattern,pattern, mask=th1)
    masked_data = cv2.bitwise_or(pattern,pattern, mask=th1)


    #cv2.imwrite(dir_tmp+"little_planet_thumb_and_pattern_test.jpg", masked_data)

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

def display_color_work(width_contact_sheet = 400, height_contact_sheet = 200):
    """The function create a mini contact sheet of the previously created color variation
    of the given little planet."""

    command_montage = "montage -label '%f' "+dir_tmp+"pano_equi_thumb.jpg \
    "+dir_tmp+"little_planet_thumb.jpg \
    "+dir_tmp+"little_planet_thumb_rotation.jpg \
    "+dir_tmp+"little_planet_thumb_BW.jpg \
    "+dir_tmp+"little_planet_thumb_BW_bin.jpg \
    "+dir_tmp+"little_planet_thumb_BW_bin_opening.jpg \
    "+dir_tmp+"little_planet_thumb_BW_equalize.jpg \
    "+dir_tmp+"little_planet_thumb_BW_equalize_bin.jpg \
    "+dir_tmp+"little_planet_thumb_BW_equalize_bin_opening.jpg \
    -tile 3x \
    -background white -geometry "+str(width_contact_sheet)+"x"+str(height_contact_sheet)+"+5+10 "+dir_tmp+"little_planet_montage.jpg"
    os.system(command_montage)
    #print command_montage
    im_lpt_montage = cv2.imread("/Users/jeremie/Pictures/tmp/little_planet_montage.jpg")
    plt.figure(figsize=(15, 15))
    plt.imshow(cv2.cvtColor(im_lpt_montage, cv2.COLOR_BGR2RGB))
    plt.xticks([])
    plt.yticks([])
    plt.show()

def display_color_work_with_pattern(little_planet_filename, width_contact_sheet = 500, height_contact_sheet = 250,
                                    file_name_reference = "thumb"):
    """The function shows the little planet together with the color pattern in
    the background.
    You need to choose which litte planet image "little_planet_filename" to use as mask.
    """
    command_montage = "montage -label '%f' "+dir_tmp+"pano_equi_"+file_name_reference+".jpg \
    "+dir_tmp+"little_planet_"+file_name_reference+".jpg " \
    +dir_tmp+little_planet_filename+ " " \
    +dir_tmp+"little_thumb_pattern_circle_1.jpg " \
    +dir_tmp+"little_thumb_pattern_circle_2.jpg " \
    +dir_tmp+"little_thumb_pattern_circle_3.jpg " \
    +dir_tmp+"little_thumb_pattern_circle_4.jpg " \
    +dir_tmp+"little_thumb_pattern_circle_5.jpg " \
    +dir_tmp+"little_thumb_pattern_circle_6.jpg " \
    +dir_tmp+"little_thumb_pattern_stripes_1.jpg " \
    +dir_tmp+"little_thumb_pattern_stripes_2.jpg " \
    +dir_tmp+"little_thumb_pattern_stripes_3.jpg " \
    +dir_tmp+"little_thumb_pattern_stripes_4.jpg " \
    +dir_tmp+"little_thumb_pattern_stripes_5.jpg " \
    +dir_tmp+"little_thumb_pattern_stripes_6.jpg \
    -tile 3x \
    -background white -geometry "+str(width_contact_sheet)+"x"+str(height_contact_sheet)+"+5+10 "+dir_tmp+"little_planet_montage_pattern.jpg"
    os.system(command_montage)
    #print command_montage
    im_lpt_montage = cv2.imread("/Users/jeremie/Pictures/tmp/little_planet_montage_pattern.jpg")
    plt.figure(figsize=(15, 15))
    plt.imshow(cv2.cvtColor(im_lpt_montage, cv2.COLOR_BGR2RGB))
    plt.xticks([])
    plt.yticks([])
    plt.show()

def export_image(little_planet_filename, export_dir, file_name_mask_reference,
                 width_contact_sheet = 400,
                 height_contact_sheet = 200,
                 file_name_suffix = "thumb"):
    """The function exports the image from the temp folder to defined export
    folder.
    """

    file_path, file_name = os.path.split(little_planet_filename)

    # check export_dir
    if os.path.exists(export_dir):
        print "The files you requested luckily exist, well done."
        print "->%s" % export_dir
        print "->%s" % file_name[5:-4]

        command_montage = "montage -label '%f' "+dir_tmp+"pano_equi_"+file_name_suffix+".jpg " \
        +dir_tmp+"little_planet_"+file_name_suffix+".jpg " \
        +dir_tmp+file_name_mask_reference+" " \
        +dir_tmp+"little_planet_"+file_name_suffix+"_BW.jpg " \
        +dir_tmp+"little_planet_"+file_name_suffix+"_BW_bin.jpg " \
        +dir_tmp+"little_planet_"+file_name_suffix+"_BW_bin_opening.jpg " \
        +dir_tmp+"little_planet_"+file_name_suffix+"_BW_equalize.jpg " \
        +dir_tmp+"little_planet_"+file_name_suffix+"_BW_equalize_bin.jpg " \
        +dir_tmp+"little_planet_"+file_name_suffix+"_BW_equalize_bin_opening.jpg " \
        +dir_tmp+"little_"+file_name_suffix+"_pattern_circle_1.jpg " \
        +dir_tmp+"little_"+file_name_suffix+"_pattern_circle_2.jpg " \
        +dir_tmp+"little_"+file_name_suffix+"_pattern_circle_3.jpg " \
        +dir_tmp+"little_"+file_name_suffix+"_pattern_circle_4.jpg " \
        +dir_tmp+"little_"+file_name_suffix+"_pattern_circle_5.jpg " \
        +dir_tmp+"little_"+file_name_suffix+"_pattern_circle_6.jpg " \
        +dir_tmp+"little_"+file_name_suffix+"_pattern_stripes_1.jpg " \
        +dir_tmp+"little_"+file_name_suffix+"_pattern_stripes_2.jpg " \
        +dir_tmp+"little_"+file_name_suffix+"_pattern_stripes_3.jpg " \
        +dir_tmp+"little_"+file_name_suffix+"_pattern_stripes_4.jpg " \
        +dir_tmp+"little_"+file_name_suffix+"_pattern_stripes_5.jpg " \
        +dir_tmp+"little_"+file_name_suffix+"_pattern_stripes_6.jpg " \
        +"-tile 3x \
        -background white -geometry "+str(width_contact_sheet)+"x" \
        +str(height_contact_sheet)+"+5+10 "+export_dir+"lpt_ContactSheet_"+file_name[5:-4]+".jpg"
        os.system(command_montage)
        #print command_montage

        # export the reference mask image
        src = dir_tmp+file_name_reference
        dst = export_dir+"lpt_ref_mask_"+file_name[5:-4]+".jpg"
        shutil.copyfile(src, dst)

        # export the different resulting images with circles and stripes
        for i in np.arange(0,6):
            src = dir_tmp+"little_"+file_name_suffix+"_pattern_circle_"+str(i+1)+".jpg"
            dst = export_dir+"lpt_pattern_circle_"+str(i+1)+"_"+file_name[5:-4]+".jpg"
            shutil.copyfile(src, dst)

            src = dir_tmp+"little_"+file_name_suffix+"_pattern_stripes_"+str(i+1)+".jpg"
            dst = export_dir+"lpt_pattern_stripes_"+str(i+1)+"_"+file_name[5:-4]+".jpg"
            shutil.copyfile(src, dst)

    else:
        print "so close and so far in the same time."
