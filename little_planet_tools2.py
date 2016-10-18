import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
import skimage
from skimage import exposure

class LittlePlanet(object):
    """A LittlePlanet is the transformation of an full spherical panoramic image
    given as en equirectangular image.

    Attibutes:
        distance: it defined how big/small it appears
        angle: it defines how much is the final LittlePlanet rotated around the
        z axis
    """

    def __init__(self, path_and_name_equirectangular, distance=300, angle=90,
                 pano_thumb_width = 1000, pano_thumb_height = 500):
        """Return a a LittlePlanet object whose distance is *distance* and
        and angle is *angle*."""

        # check if the image has a ratio of 2 and if not print a warning
        file_path = os.path.dirname(path_and_name_equirectangular)
        file_name = os.path.basename(path_and_name_equirectangular)
        file_path, file_name = os.path.split(path_and_name_equirectangular)

        self.pano_equirectangular_name = file_name
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
            print "Ratio of image size %1.0fx%1.0f is %1.0f" % (self.pano_width,
                                                                self.pano_height,
                                                                ratio_width_height)
            self.pano_thumb_width = pano_thumb_width
            self.pano_thumb_height = pano_thumb_height

        else:
            print "You should do something with your life."

        # resize image to get a thumb faster manageable version
        self.im_pano_equirectangular_thumb = cv2.resize(self.im_pano_equirectangular,
                                                        (self.pano_thumb_width,self.pano_thumb_height),
                                                        interpolation = cv2.INTER_CUBIC)
        # save thumb as a jpg image somewhere
        self.im_pano_equirectangular_thumb = cv2.cvtColor(self.im_pano_equirectangular_thumb,
                                                          cv2.COLOR_BGR2RGB)
        cv2.imwrite("/Users/jeremie/Pictures/tmp/pano_equi_thumb.jpg", self.im_pano_equirectangular_thumb)

        # basic parameters for the config file
        self.distance = distance
        self.r_angle = -90
        self.p_angle = angle
        self.y_angle = -90
        self.var_output_width = self.pano_thumb_width
        self.var_output_height = self.pano_thumb_height
        self.output_file_name = "little_planet_thumb.jpg"

        # with _init__ a config file should be automatically created
        create_pto_config_file(path_and_name_equirectangular,
                            output_file_name = "template333",
                            output_width = self.pano_thumb_width,
                            output_height = self.pano_thumb_height,
                            r_angle = self.r_angle,
                            p_angle = self.p_angle,
                            y_angle = self.y_angle,
                            output_distance = self.distance)

        # save as above we generate directly a little planet
        create_little_planet(path_and_name_equirectangular,
                            config_file_pto = "template333.pto",
                            litte_planet_name = self.output_file_name)

#    def update_thumb(self, new_width, new_height):
#        """The function updates the size of the thumb image.
#        """
#        self.pano_thumb_width = new_width
#        self.pano_thumb_height = new_height
#        del(self.pano_equirectangular_thumb)
#        self.pano_equirectangular_thumb = cv2.resize(self.im_pano_equirectangular,
#                                    (self.pano_thumb_width, self.pano_thumb_height),
#                                    interpolation = cv2.INTER_CUBIC)
#        print np.shape(self.pano_equirectangular_thumb)
#        os.remove("/Users/jeremie/Pictures/tmp/pano_equi_thumb.jpg")
#        cv2.imwrite("/Users/jeremie/Pictures/tmp/pano_equi_thumb.jpg", self.im_pano_equirectangular_thumb)

# A few functions not specially linked to the class
def display_image(image_data, image_name):
    """Thte function display an image, panorama image of little planet.
    """
    fig = plt.figure(figsize=(8,8))
    plt.imshow(image_data)
    plt.xticks([])
    plt.yticks([])
    plt.title(image_name)
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
    print img_size, np.shape(img_size)
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
                         litte_planet_name = "little_planet_thumb"):
    """The function will call the funcion create_pto_config_file to be able to
    create the final little planet."""

    # apply command
    command_line_for_Hugin = "/Applications/Hugin/HuginTools/nona -o /Users/jeremie/Pictures/tmp/imLittlePlanet33 -m TIFF %s %s" % (config_file_pto,path_and_name_equirectangular)
    print command_line_for_Hugin
    os.system(command_line_for_Hugin)

    # convert the TIFF file to jpg
    im_little_planet_tiff = cv2.imread("/Users/jeremie/Pictures/tmp/imLittlePlanet33.tif")
    cv2.imwrite("/Users/jeremie/Pictures/tmp/little_planet_thumb.jpg",im_little_planet_tiff)
    # remove the TIFF image
    os.remove("/Users/jeremie/Pictures/tmp/imLittlePlanet33.tif")

def do_color_work(little_planet_name):
    """The function applies some color transformation on the resulting little planet
    image that has been created."""

    # convert to BW to bin
    im_little_planet = cv2.imread(little_planet_name)
    im_little_planet_BW = cv2.cvtColor(im_little_planet, cv2.COLOR_BGR2GRAY)
    cv2.imwrite(little_planet_name[0:-4]+"_BW.jpg", im_little_planet_BW)

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

def display_color_work(width_contact_sheet = 400, height_contact_sheet = 200):
    """The function create a mini contact sheet of the previously created color variation
    of the given little planet."""
    dir_tmp = "/Users/jeremie/Pictures/tmp/"

    command_montage = "montage -label '%f' "+dir_tmp+"pano_equi_thumb.jpg \
    "+dir_tmp+"little_planet_thumb.jpg \
    "+dir_tmp+"little_planet_thumb_BW.jpg \
    "+dir_tmp+"little_planet_thumb_BW_bin.jpg \
    "+dir_tmp+"little_planet_thumb_BW_equalize.jpg \
    "+dir_tmp+"little_planet_thumb_BW_equalize_bin.jpg -tile 2x \
    -background white -geometry "+str(width_contact_sheet)+"x"+str(height_contact_sheet)+"+5+10 "+dir_tmp+"little_planet_montage.jpg"
    os.system(command_montage)
    #print command_montage
    im_lpt_montage = cv2.imread("/Users/jeremie/Pictures/tmp/little_planet_montage.jpg")
    plt.figure(figsize=(15, 15))
    plt.imshow(cv2.cvtColor(im_lpt_montage, cv2.COLOR_BGR2RGB))
    plt.xticks([])
    plt.yticks([])
    plt.show()
