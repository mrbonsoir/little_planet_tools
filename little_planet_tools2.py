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

    def __init__(self, path_and_name_equirectangular, distance=300,
                r_angle = -90, angle=90, y_angle = -90,
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
        self.r_angle = r_angle #-90
        self.p_angle = angle
        self.y_angle = y_angle #-90
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
        """The function applies some color transformations on the resulting
        little planet image that has been created."""

        # convert to BW to bin
        im_little_planet = cv2.imread(little_planet_name)
        im_little_planet_BW = cv2.cvtColor(im_little_planet, cv2.COLOR_BGR2GRAY)
        cv2.imwrite(little_planet_name[0:-4]+"_BW.jpg", im_little_planet_BW)
        self.im_little_planet_BW = im_little_planet_BW

        im_little_planet_BW_bin = cv2.GaussianBlur(im_little_planet_BW,(5,5),0)
        ret,th1 = cv2.threshold(im_little_planet_BW_bin,127,255,cv2.THRESH_BINARY)
        cv2.imwrite(little_planet_name[0:-4]+"_BW_bin.jpg",th1)

        # do some morphological operation on the binaray image
        kernel = np.ones((5,5),np.uint8)
        opening_bin = cv2.morphologyEx(th1, cv2.MORPH_OPEN, kernel)
        cv2.imwrite(little_planet_name[0:-4]+"_BW_bin_opening.jpg",opening_bin)

        # convert BW to BW equalize to bin
        im_little_planet_BW_equalize = cv2.equalizeHist(im_little_planet_BW)
        cv2.imwrite(little_planet_name[0:-4]+"_BW_equalize.jpg",im_little_planet_BW_equalize)

        # Convert BW equalize to binary Otsu's thresholding after Gaussian filtering
        im_little_planet_BW_equalize_bin = cv2.GaussianBlur(im_little_planet_BW_equalize,(5,5),0)
        ret,th11 = cv2.threshold(im_little_planet_BW_equalize_bin,127,255,cv2.THRESH_BINARY)
        cv2.imwrite(little_planet_name[0:-4]+"_BW_equalize_bin.jpg",th11)

        # do some morphological operation on the binaray image
        opening_equalize_bin = cv2.morphologyEx(th11, cv2.MORPH_OPEN, kernel)
        cv2.imwrite(little_planet_name[0:-4]+"_BW_equalize_bin_opening.jpg",opening_equalize_bin)

        # do a pixelized version of the little planet
        # resize to 16 x 32

        #im_LP_pixel = pixelize_image(im_little_planet, pixel_width = 2, pixel_height = 1)
        #cv2.imwrite(little_planet_name[0:-4]+"_pixel0.jpg",im_LP_pixel)

        im_LP_pixel = pixelize_image(im_little_planet, pixel_width = 4, pixel_height = 2)
        cv2.imwrite(little_planet_name[0:-4]+"_pixel1.jpg",im_LP_pixel)

        im_LP_pixel = pixelize_image(im_little_planet, pixel_width = 8, pixel_height = 4)
        cv2.imwrite(little_planet_name[0:-4]+"_pixel2.jpg",im_LP_pixel)

        im_LP_pixel = pixelize_image(im_little_planet, pixel_width = 32, pixel_height = 16)
        cv2.imwrite(little_planet_name[0:-4]+"_pixel3.jpg",im_LP_pixel)


        im_LP_pixel = pixelize_image(im_little_planet, pixel_width = 64, pixel_height = 32)
        cv2.imwrite(little_planet_name[0:-4]+"_pixel4.jpg",im_LP_pixel)
        """

        im_LP_pixel = pixelize_image(im_little_planet, pixel_width = 128, pixel_height = 64)
        cv2.imwrite(little_planet_name[0:-4]+"_pixel5.jpg",im_LP_pixel)

        im_LP_pixel = pixelize_image(im_little_planet, pixel_width = 256, pixel_height = 128)
        cv2.imwrite(little_planet_name[0:-4]+"_pixel6.jpg",im_LP_pixel)

        im_LP_pixel = pixelize_image(im_little_planet, pixel_width = 512, pixel_height = 256)
        cv2.imwrite(little_planet_name[0:-4]+"_pixel7.jpg",im_LP_pixel)

        im_LP_pixel = pixelize_image(im_little_planet, pixel_width = 1024, pixel_height = 512)
        cv2.imwrite(little_planet_name[0:-4]+"_pixel7.jpg",im_LP_pixel)

        im_LP_pixel = pixelize_image(im_little_planet, pixel_width = 2048, pixel_height = 1024)
        cv2.imwrite(little_planet_name[0:-4]+"_pixel8.jpg",im_LP_pixel)
        """

        # NOT VERY NECESSARY TO HAVE THE ROTATION IMAGE
        # rotation of n degrees the color and BW images
        #im_little_planet_rotation = rotate_image(im_little_planet)
        #cv2.imwrite(little_planet_name[0:-4]+"_rotation.jpg",im_little_planet_rotation)

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

def pixelize_image(image_rgb, pixel_width = 4, pixel_height = 2):
    """The function create four versions of a pixelized image of the little
    planet image and save them of course.
    """
    image_shape = np.shape(image_rgb)
    im_little_planet_small =  cv2.resize(image_rgb,
                              (pixel_width, pixel_height),
                              interpolation = cv2.INTER_CUBIC)

    # resize to original size
    im_LP_pixel = cv2.resize(im_little_planet_small,
                             (image_shape[1], image_shape[0]),
                             interpolation = cv2.INTER_NEAREST)

    return im_LP_pixel

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

def display_color_work(width_contact_sheet = 400, height_contact_sheet = 200, file_name_suffix = "thumb"):
    """The function create a mini contact sheet of the previously created color variation
    of the given little planet."""

    command_montage = "montage -label '%f' "+dir_tmp+"pano_equi_"+file_name_suffix+".jpg \
    "+dir_tmp+"little_planet_"+file_name_suffix+"_pixel1.jpg \
    "+dir_tmp+"little_planet_"+file_name_suffix+"_pixel2.jpg \
    "+dir_tmp+"little_planet_"+file_name_suffix+".jpg \
    "+dir_tmp+"little_planet_"+file_name_suffix+"_pixel4.jpg \
    "+dir_tmp+"little_planet_"+file_name_suffix+"_pixel3.jpg \
    "+dir_tmp+"little_planet_"+file_name_suffix+"_BW.jpg \
    "+dir_tmp+"little_planet_"+file_name_suffix+"_BW_bin.jpg \
    "+dir_tmp+"little_planet_"+file_name_suffix+"_BW_bin_opening.jpg \
    "+dir_tmp+"little_planet_"+file_name_suffix+"_BW_equalize.jpg \
    "+dir_tmp+"little_planet_"+file_name_suffix+"_BW_equalize_bin.jpg \
    "+dir_tmp+"little_planet_"+file_name_suffix+"_BW_equalize_bin_opening.jpg \
    -tile 3x -background black -fill white \
    -geometry "+str(width_contact_sheet)+"x"+str(height_contact_sheet)+"+5+10 "+dir_tmp+"little_planet_montage.jpg"
    os.system(command_montage)
    #print command_montage
    im_lpt_montage = cv2.imread("/Users/jeremie/Pictures/tmp/little_planet_montage.jpg")
    plt.figure(figsize=(15, 15))
    plt.imshow(cv2.cvtColor(im_lpt_montage, cv2.COLOR_BGR2RGB))
    plt.xticks([])
    plt.yticks([])
    plt.show()

    # For now I don't really need this multiple pixle image rendering
    """
    command_montage = "montage "+dir_tmp+"little_planet_"+file_name_suffix+"_pixel[0-8].jpg \
    -tile 3x3 -background white \
    -geometry 2000x1000+10+10 "+dir_tmp+"little_planet_pixel_montage.jpg"

    os.system(command_montage)
    #print command_montage
    im_lpt_montage = cv2.imread("/Users/jeremie/Pictures/tmp/little_planet_pixel_montage.jpg")
    plt.figure(figsize=(15, 15))
    plt.imshow(cv2.cvtColor(im_lpt_montage, cv2.COLOR_BGR2RGB))
    plt.xticks([])
    plt.yticks([])
    plt.show()
    """

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
