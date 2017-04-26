import numpy as np
import cv2

def fun_redorder_pixel_by_raw(imRGB):
    """The function reorder the image by sorting each raw by Hue"""

    imHLS = cv2.cvtColor(imRGB,cv2.COLOR_RGB2HLS)
    imHLS = cv2.cvtColor(imRGB,cv2.COLOR_RGB2HSV)

    imRGB_reorder = np.zeros((np.shape(imRGB)[0],
                              np.shape(imRGB)[1],
                              np.shape(imRGB)[2]),
                            dtype = imRGB.dtype)

    for r in np.arange(np.shape(imRGB)[0]):
        raw = np.argsort(imHLS[r,:,2])
        imRGB_reorder[r,:,:] = imRGB[r,raw,:]

    return imRGB_reorder


def fun_redorder_pixel_by_col(imRGB):
    """The function reorder the image by sorting each col by Hue"""

    imHLS = cv2.cvtColor(imRGB,cv2.COLOR_RGB2HLS)
    imHLS = cv2.cvtColor(imRGB,cv2.COLOR_RGB2HSV)

    imRGB_reorder = np.zeros((np.shape(imRGB)[0],
                              np.shape(imRGB)[1],
                              np.shape(imRGB)[2]),
                            dtype = imRGB.dtype)

    for c in np.arange(np.shape(imRGB)[1]):
        col = np.argsort(imHLS[:,c,2])
        imRGB_reorder[:,c,:] = imRGB[col,c,:]

    return imRGB_reorder


def fun_reorder_pixel_by_hue(imRGB):
    """All the pixels of the image are taken into account"""

    imHLS = cv2.cvtColor(imRGB,cv2.COLOR_RGB2HLS)
    imHLS = cv2.cvtColor(imRGB,cv2.COLOR_RGB2HSV)

    imRGB_reorder = np.zeros((np.shape(imRGB)[0],
                              np.shape(imRGB)[1],
                              np.shape(imRGB)[2]),
                            dtype = imRGB.dtype)

    # reorder by Saturation
    imS = imHLS[:,:,2].flatten()
    new_order = np.argsort(imS)

    for i in np.arange(3):
        data_single_channel  = imRGB[:,:,i].flatten()
        data_single_channel  = data_single_channel[new_order]
        imRGB_reorder[:,:,i] = data_single_channel.reshape(np.shape(imRGB)[0], np.shape(imRGB)[1])

    return imRGB_reorder

    """
    imHLS2 = cv2.cvtColor(imRGB_reorder, cv2.COLOR_RGB2HLS)

    # reorder by Luminance
    imL = imHLS2[:,:,1].flatten()
    new_order2 = np.argsort(imL)
    imRGB_reorder2 = np.zeros((np.shape(imRGB)[0], np.shape(imRGB)[1], np.shape(imRGB)[2]), dtype = imRGB.dtype)

    for i in np.arange(3):
        data_single_channel   = imRGB_reorder[:,:,i].flatten()
        data_single_channel   = data_single_channel[new_order2]
        imRGB_reorder2[:,:,i] = data_single_channel.reshape(np.shape(imRGB)[0], np.shape(imRGB)[1])

    imHLS3 = cv2.cvtColor(imRGB_reorder2, cv2.COLOR_RGB2HLS)

    # reorder by Hue
    imH = imHLS3[:,:,0].flatten()
    new_order3 = np.argsort(imH)

    imRGB_reorder3 = np.zeros((np.shape(imRGB)[0], np.shape(imRGB)[1], np.shape(imRGB)[2]), dtype = imRGB.dtype)

    for i in np.arange(3):
        data_single_channel = imRGB_reorder2[:,:,i].flatten()
        data_single_channel = data_single_channel[new_order3[::-1]]
        imRGB_reorder3[:,:,i] = data_single_channel.reshape(np.shape(imRGB)[0],
                                                   np.shape(imRGB)[1])

    return imRGB_reorder2
    """
