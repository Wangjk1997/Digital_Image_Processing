import dippykit as dip
import numpy as np

def scale(img: np.ndarray):
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            if img[i, j] > 1:
                img[i, j] = 1
            if img[i, j] < 0:
                img[i, j] = 0
    return img


img = dip.im_read('barbara.png')
Gaussian = 1/16*np.array([[1,2,1],[2,4,2],[1,2,1]])


img1 = dip.convolve2d(img,Gaussian,mode='same')
img2 = dip.convolve2d(img1,Gaussian,mode='same')
img3 = dip.convolve2d(img2,Gaussian,mode='same')

Lap_kernel = np.array([[0,1,0],[1,-4,1],[0,1,0]])
ELap_kernel = np.array([[1,1,1],[1,-8,1],[1,1,1]])

Lap1 = dip.convolve2d(img1,Lap_kernel,mode='same')
Lap2 = dip.convolve2d(img2,Lap_kernel,mode='same')
Lap3 = dip.convolve2d(img3,Lap_kernel,mode='same')

ELap1 = dip.convolve2d(img1,ELap_kernel,mode='same')
ELap2 = dip.convolve2d(img2,ELap_kernel,mode='same')
ELap3 = dip.convolve2d(img3,ELap_kernel,mode='same')

sharpen1 = scale(img1 - Lap1)
print('sharpen1')
print(dip.PSNR(img,sharpen1))
sharpen2 = scale(img2 - Lap2)
print('sharpen2')
print(dip.PSNR(img,sharpen2))
sharpen3 = scale(img3 - Lap3)
print('sharpen3')
print(dip.PSNR(img,sharpen3))

Esharpen1 = scale(img1 - ELap1)
print('Esharpen1')
print(dip.PSNR(img,Esharpen1))
Esharpen2 = scale(img2 - ELap2)
print('Esharpen2')
print(dip.PSNR(img,Esharpen2))
Esharpen3 = scale(img3 - ELap3)
print('Esharpen3')
print(dip.PSNR(img,Esharpen3))

dip.figure(1)
dip.subplot(1,3,1)
dip.imshow(img1,'gray')
dip.title('Minor Distortion')
dip.subplot(1,3,2)
dip.imshow(img2,'gray')
dip.title('Mild Distortion')
dip.subplot(1,3,3)
dip.imshow(img3,'gray')
dip.title('Severe Distortion')

dip.figure(2)
dip.subplot(1,3,1)
dip.imshow(Lap1,'gray')
dip.title('Minor Distortion')
dip.subplot(1,3,2)
dip.imshow(Lap2,'gray')
dip.title('Mild Distortion')
dip.subplot(1,3,3)
dip.imshow(Lap3,'gray')
dip.title('Severe Distortion')

dip.figure(3)
dip.subplot(1,3,1)
dip.imshow(ELap1,'gray')
dip.title('Minor Distortion')
dip.subplot(1,3,2)
dip.imshow(ELap2,'gray')
dip.title('Mild Distortion')
dip.subplot(1,3,3)
dip.imshow(ELap3,'gray')
dip.title('Severe Distortion')

dip.figure(4)
dip.subplot(1,3,1)
dip.imshow(sharpen1,'gray')
dip.title('Minor Distortion')
dip.subplot(1,3,2)
dip.imshow(sharpen2,'gray')
dip.title('Mild Distortion')
dip.subplot(1,3,3)
dip.imshow(sharpen3,'gray')
dip.title('Severe Distortion')

dip.figure(5)
dip.subplot(1,3,1)
dip.imshow(Esharpen1,'gray')
dip.title('Minor Distortion')
dip.subplot(1,3,2)
dip.imshow(Esharpen2,'gray')
dip.title('Mild Distortion')
dip.subplot(1,3,3)
dip.imshow(Esharpen3,'gray')
dip.title('Severe Distortion')

dip.show()