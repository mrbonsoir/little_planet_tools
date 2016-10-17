from PIL import Image
import os
import cv2
import numpy as np


def fun_create_little_planet(pano_equirectangular_name, config_file_pto, litte_planet_name):
    '''The function create a little planet.
    
    In:
        - pano_equirectangular_name (str): path and name of the image to process
        - config_file_pto (str): name of the config file for Hugin 
        - litte_planet_name
    '''

    # apply command
    commandLineForHugin = "../../../Applications/HuginTools/nona -o %s -m TIFF %s %s" % (litte_planet_name, pano_equirectangular_name, config_file_pto)
    os.system(commandLineForHugin)

def fun_transform_pano2littleplanet(pano_equirectangular_name, config_file_name = "template33", pano_little_planet_name = "pano_little_planet.jpg", destination_head_path="/"):
    """The function can be called after the config file has been created.
    It return the image as an numpy array as usual and save the little planet image as jpg file.
    """

    #print pano_equirectangular_name

    # get image path
    head_path, tail_path = os.path.split(pano_equirectangular_name)

    # get the destination image path
    if destination_head_path != "/":
        # new path MEANING one already created that can located anywhere
        destination_head_path = destination_head_path
        if destination_head_path[-1] != "/":
            destination_head_path = destination_head_path+"/"
    else:
        # we copy the image where the function is called
        destination_head_path == os.getcwd()+"/"

    #print "Destination path : %s" % destination_head_path

    # apply command
    if config_file_name[-4:] != ".pto":
        config_file_name = config_file_name+".pto"
    
    command_line_for_Hugin = "/Applications/Hugin/HuginTools/nona -o imLittlePlanet33 -m TIFF %s %s" % (config_file_name, pano_equirectangular_name)
    #print command_line_for_Hugin
    os.system(command_line_for_Hugin)

    # convert the create TIF image into jpg
    im_little_planet_tiff = cv2.imread(os.getcwd()+"/imLittlePlanet33.tif")
    cv2.imwrite(destination_head_path+"/"+pano_little_planet_name, im_little_planet_tiff)

def fun_create_config_for_littleplanet(pano_equirectangular_name, output_file_name = "template33",
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

    imPanoEqui   = Image.open(pano_equirectangular_name)
    image_width  = imPanoEqui.size[0]
    image_height = imPanoEqui.size[1]

    #hugin_ptoversion 2
    f = open(output_file_name+'.pto', 'w')
    f.write('# hugin project file\n')
    f.write('p f4 w'+str(output_width)+' h'+str(output_height)+' v'+str(output_distance)+' E0 R0 n"TIFF_m c:LZW r:CROP"\n')
    f.write('m g1 i0 f0 m2 p0.00784314\n')

    # image lines
    #-hugin  cropFactor=1
    f.write('i w'+str(image_width)+' h'+str(image_height)+' f4 v360 Ra0 Rb0 Rc0 Rd0 Re0 Eev0 Er1 Eb1 r'+str(r_angle)+' p'+str(p_angle)+' y'+str(y_angle)+' TrX0 TrY0 TrZ0 Tpy0 Tpp0 j0 a0 b0 c0 d0 e0 g0 t0 Va1 Vb0 Vc0 Vd0 Vx0 Vy0  Vm5 n"'+pano_equirectangular_name+'"\n')

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

def fun_save_all_images(pano_little_planet_name, destination_path, img, img_gray, img_bin_gray_binary, img_bin_inverse):
    """
    The function saves all images in rectangle and square format.
    Eventually it also create a montage of the square image versions.
    """
    print "image name is %s" % pano_little_planet_name
    ss = np.shape(img)
    print destination_path
    cv2.imwrite(destination_path+"/"+"lp_r0_"+pano_little_planet_name, img)
    cv2.imwrite(destination_path+"/"+"lp_s0_"+pano_little_planet_name, img[:,np.round(ss[0] / 2):np.round(ss[0] / 2)+ss[0],:])

    # save gray version of the image
    cv2.imwrite(destination_path+"/"+"lp_r1_"+pano_little_planet_name, img_gray)
    cv2.imwrite(destination_path+"/"+"lp_s1_"+pano_little_planet_name, img_gray[:,np.round(ss[0] / 2):np.round(ss[0] / 2)+ss[0]])
    
    # save the inverse mask image
    img_bin_gray_binary = img_bin_gray_binary.astype(int) * 255
    cv2.imwrite(destination_path+"/"+"lp_r2_"+pano_little_planet_name, np.float32(img_bin_gray_binary))
    cv2.imwrite(destination_path+"/"+"lp_s2_"+pano_little_planet_name, 
                                                        np.float32(img_bin_gray_binary[:,np.round(ss[0] / 2):np.round(ss[0] / 2)+ss[0]]))
    
    # save the inverse mask image
    img_bin_inverse = img_bin_inverse.astype(int) * 255
    cv2.imwrite(destination_path+"/"+"lp_r3_"+pano_little_planet_name, np.float32(img_bin_inverse))
    cv2.imwrite(destination_path+"/"+"lp_s3_"+pano_little_planet_name, 
                                                        np.float32(img_bin_inverse[:,np.round(ss[0] / 2):np.round(ss[0] / 2)+ss[0]]))

    del img, img_gray, img_bin_gray_binary, img_bin_inverse

    # create the mosaic of rectangle images
    command_montage = "montage -tile 1x4 "+destination_path+"/lp_r0_"+pano_little_planet_name+" "+destination_path+"/lp_r1_"+pano_little_planet_name+" "+destination_path+"/lp_r2_"+pano_little_planet_name+" "+destination_path+"/lp_r3_"+pano_little_planet_name+" -background white -border 5 -bordercolor fuchsia -geometry +5+5 "+destination_path+"/imMontage4Rect_Ver_"+pano_little_planet_name 
    os.system(command_montage)

    # create the mosaic of square images
    command_montage = "montage -tile 2x2 "+destination_path+"/lp_s0_"+pano_little_planet_name+" "+destination_path+"/lp_s1_"+pano_little_planet_name+" "+destination_path+"/lp_s2_"+pano_little_planet_name+" "+destination_path+"/lp_s3_"+pano_little_planet_name+" -background white -border 5 -bordercolor fuchsia -geometry +5+5 "+destination_path+"/imMontage4Squares_"+pano_little_planet_name 
    os.system(command_montage)

    command_montage = "montage -tile 1x4 "+destination_path+"/lp_s0_"+pano_little_planet_name+" "+destination_path+"/lp_s1_"+pano_little_planet_name+" "+destination_path+"/lp_s2_"+pano_little_planet_name+" "+destination_path+"/lp_s3_"+pano_little_planet_name+" -background white -border 5 -bordercolor fuchsia -geometry +5+5 "+destination_path+"/imMontage4Squares_Ver_"+pano_little_planet_name 
    os.system(command_montage)


def fun_create_frame_for_lp_animation(im_pano_equirectangular_name,
                                      destination_head_path,
                                      var_p_angle=90,
                                      val_angle=45,
                                      var_output_width=1024,
                                      var_output_height=512,
                                      var_output_distance=330):
    """
    The function creates several images/frame with only the rotation angle value
    changing, such that we can have the illusion the planet is turning around.

    It generates a new configuration file for each frame.

    IN:
        im_pano_equirectangular_name (str):  the name of the equirectangular
        destination_head_path (str): the path where to store the little_planet
        val_angle (int): 45 if 45 then angle value are 0 / 45 / 90 / ...
        output_width (int): 1024
        output_height (int): 512
        output_distance (int): 330
    OUT:


    """
    print "img name: %s " % im_pano_equirectangular_name
    print "img dest: %s" % destination_head_path
    print "angle start: %1.0f" % var_p_angle
    print "angle step %1.0f" % val_angle
    print "output width %1.0f" % var_output_width
    print "output height %1.0f" % var_output_height
    print "output distance %1.0f" % var_output_distance

    c = 0
    for angle_value in np.arange(0, 360, val_angle):
        little_planet_name = 'frame_' + str(c).zfill(3)
        print angle_value,  # little_planet_name

        # config file
        fun_create_config_for_littleplanet(im_pano_equirectangular_name,
                                               output_file_name="config33_animation",
                                               output_width=var_output_width,
                                               output_height=var_output_height,
                                               r_angle=-90, p_angle=var_p_angle + angle_value, y_angle=-90,
                                               output_distance=var_output_distance)

        # apply config file
        fun_transform_pano2littleplanet(im_pano_equirectangular_name,
                                        config_file_name="config33_animation",
                                        pano_little_planet_name=little_planet_name + ".jpg",
                                        destination_head_path=destination_head_path)
        c += 1
    print "\nNow we are done. Good job you did."



def fun_create_frame_for_lp_animation_with_spiral(im_pano_equirectangular_name,
                                      destination_head_path,
                                      var_p_angle=90,
                                      val_angle=45,
                                      var_output_width=1024,
                                      var_output_height=512,
                                      var_output_distance= np.array([360, 330]),
                                      start_index_frame = 0):
    """
    The function does like the one above but this time with a parameter for the distance as well.

    It generates a new configuration file for each frame.

    IN:
        im_pano_equirectangular_name (str):  the name of the equirectangular
        destination_head_path (str): the path where to store the little_planet
        val_angle (int): 45 if 45 then angle value are 0 / 45 / 90 / ...
        output_width (int): 1024
        output_height (int): 512
        output_distance (int): [360,330] for each angle will be a corresponding distance from 360 to
                                the output_distance or the other way around
    OUT:
        frame are save in the destination head path.

    """
    print "img name: %s " % im_pano_equirectangular_name
    print "dest img: %s" % destination_head_path
    print "angle start: %1.0f" % var_p_angle
    print "angle step %1.0f" % val_angle
    print "output width %1.0f" % var_output_width
    print "output height %1.0f" % var_output_height
    print "output distance %1.0f -> %1.0f" % (var_output_distance[0], var_output_distance[1])

    c = start_index_frame

    vec_angle_value = np.arange(0, 360, val_angle)
    vec_distance_value = np.linspace(var_output_distance[0], var_output_distance[1], len(vec_angle_value))

    for angle_value, distance_value in zip(vec_angle_value, vec_distance_value):
        little_planet_name = 'frame_' + str(c).zfill(3)
        #print "angle %1.0f distance %1.0f" % (angle_value,  distance_value)

        # config file
        fun_create_config_for_littleplanet(im_pano_equirectangular_name,
                                               output_file_name="config33_animation",
                                               output_width=var_output_width,
                                               output_height=var_output_height,
                                               r_angle=-90, p_angle=var_p_angle + angle_value, y_angle=-90,
                                               output_distance=distance_value)

        # apply config file
        fun_transform_pano2littleplanet(im_pano_equirectangular_name,
                                        config_file_name="config33_animation",
                                        pano_little_planet_name=little_planet_name + ".jpg",
                                        destination_head_path=destination_head_path)
        c += 1
        
    print "angle %1.0f distance %1.0f" % (angle_value, distance_value)
    print "\nNow we are done. Good job you did."
