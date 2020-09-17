import dippykit as dip
import numpy as np


def zoomAndshrink(input_IMG,r_row,r_col,theta):
    M1 = np.array([[r_col,0],[0,r_row]])
    M2 = np.array([[np.cos(theta),np.sin(theta)],[-np.sin(theta),np.cos(theta)]])
    tmp1 = dip.resample(input_IMG,M1,interp='bicubic')
    tmp2 = dip.resample(tmp1,M2,interp='bicubic')
    return tmp2

#inverse transform
def zoomAndshrink_inv(input_IMG,r_row,r_col,theta,original_size):
    M1 = np.array([[r_col,0],[0,r_row]])
    M2 = np.array([[np.cos(theta),np.sin(theta)],[-np.sin(theta),np.cos(theta)]])
    tmp1 = dip.resample(input_IMG,M2,interp='bicubic')
    tmp2 = dip.resample(tmp1,M1,interp='bicubic')

#adjusting the size of inverse transform
    row = original_size[0]
    col = original_size[1]
    if tmp2.shape[0] < row:
        error_row = row - tmp2.shape[0]
        tmp2 = np.pad(tmp2,((error_row,error_row),(0,0),(0,0)),'constant',constant_values = (0,0))
    if tmp2.shape[1] < col:
        error_col = col - tmp2.shape[1]
        tmp2 = np.pad(tmp2,((0,0),(error_col,error_col),(0,0)),'constant',constant_values = (0,0))
    tmp3 = tmp2[   int(tmp2.shape[0] / 2 - 1 - row / 2 + 1):int(tmp2.shape[0] / 2 + row / 2),
                   int(tmp2.shape[1] / 2 - 1 - col / 2 + 1):int(tmp2.shape[1] / 2 + col / 2),
                :]
    tmp4 = np.clip(tmp3,0,1)
    return tmp4

#example
filename = 'images/airplane.jpg'
im = dip.im_read(filename)
#downSampling
down_IMG = zoomAndshrink(im,2.5,1.0/1.7,27.5/180.0*np.pi)
dip.figure(1)
dip.imshow(down_IMG)

#upSampling
up_IMG = zoomAndshrink_inv(down_IMG,0.4,1.7,-27.5/180.0*np.pi,im.shape)
dip.figure(2)
dip.imshow(up_IMG)

#show difference
dip.figure(3)
dip.imshow(im-up_IMG)

#show PSNR
print(dip.PSNR(im,up_IMG))
dip.show()