# 

import numpy as np
import imageio as io


# add functions belwo
def create_pto_config_file(path_and_name_equirectangular_image, output_file_name = "template33",
                           output_width = 3000, output_height = 1500,
                           r_angle = 0, p_angle = 90, y_angle = 0,
                           output_distance = 300):
    """This function create a pto file for hugin that can be used with nona.

    It uses a template in which we can change several parameters.
    What doesn't change is that we start from an equirectangular panorama that we remap to
    a spherical stereographic. 

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
    
	output:
		nothing is returned but a config file template33.pto is created.

    """

    # get the size parameters from the pano equirectangular
    im_pano_equirectangular = io.v2.imread(path_and_name_equirectangular_image)
    img_size = np.shape(im_pano_equirectangular)
    image_width  = img_size[1]
    image_height = img_size[0]

    #hugin_ptoversion 2
    f = open(output_file_name+'.pto', 'w')
    f.write('# hugin project file\n')
    f.write('p f4 w'+str(output_width)+' h'+str(output_height)+' v'+str(output_distance)+' E0 R0 n"TIFF_m c:LZW r:CROP"\n')
    f.write('m g1 i0 f0 m2 p0.00784314\n')

    # image lines
    #-hugin  cropFactor=1
    f.write('i w'+str(image_width)+' h'+str(image_height)+' f4 v360 Ra0 Rb0 Rc0 Rd0 Re0 Eev0 Er1 Eb1 r'+str(r_angle)+' p'+str(p_angle)+' y'+str(y_angle)+' TrX0 TrY0 TrZ0 Tpy0 Tpp0 j0 a0 b0 c0 d0 e0 g0 t0 Va1 Vb0 Vc0 Vd0 Vx0 Vy0  Vm5 n"'+path_and_name_equirectangular_image+'"\n')

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

    return 0