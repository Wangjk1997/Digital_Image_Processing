import dippykit as dip
import numpy as np

f = dip.im_read('cameraman.tif')
f = dip.float_to_im(f,8)
g = dip.image_noise(f, 'gaussian' , var=(20**2))

def filter(h, gext:np.ndarray, m:int, n:int, L:int ):
    nominator = 0
    denominator = 0
    for k in range(-L, L+1):
        for l in range(-L, L+1):
            nominator = nominator + h(k,l)*gext[n+L-k,m+L-l]
            denominator = denominator + h(k,l)
    result = nominator/denominator
    return result

#Gaussian Filter
sigma1 = 1
L = 3
h1 = lambda k,l: np.exp(-(k*k + l*l)/(2*sigma1*sigma1))
gext = np.pad(g,(L,L), mode='symmetric')
output1 = np.zeros(g.shape)
for m in range(output1.shape[1]):
    for n in range(output1.shape[0]):
        output1[n,m] = filter(h1, gext, m, n, L)
print('MSE1')
print(dip.MSE(f,output1))
#Range (or Sigma) Filter
rho1 = 40
L = 3
output2 = np.zeros(g.shape)
gext = np.pad(g,(L,L), mode='symmetric')
for m in range(output2.shape[1]):
    for n in range(output2.shape[0]):
        nominator = 0
        denominator = 0
        for k in range(-L, L + 1):
            for l in range(-L, L + 1):
                nkl = float(gext[n + L - k, m + L - l])
                nnm = float(gext[n+L, m+L])
                nominator = nominator + np.exp(-1/2*(((nkl-nnm)/rho1)**2)) * nkl
                denominator = denominator + np.exp(-1/2*(((nkl-nnm)/rho1)**2))
        output2[n, m] = nominator/denominator
print('MSE2')
print(dip.MSE(f,output2))
#Bilateral Filter
sigma2 = 2
rho2 = 50
L = 6
output3 = np.zeros(g.shape)
gext = np.pad(g,(L,L), mode='symmetric')
h3 = lambda k,l: np.exp(-(k*k + l*l)/(2*sigma2*sigma2))
for m in range(output3.shape[1]):
    for n in range(output3.shape[0]):
        nominator = 0
        denominator = 0
        for k in range(-L, L + 1):
            for l in range(-L, L + 1):
                nkl = float(gext[n + L - k, m + L - l])
                nnm = float(gext[n+L, m+L])
                nominator = nominator + h3(k,l)*np.exp(-1/2*(((nkl-nnm)/rho2)**2)) * nkl
                denominator = denominator + h3(k,l)*np.exp(-1/2*(((nkl-nnm)/rho2)**2))
        output3[n, m] = nominator/denominator
    print(m)
print('MSE3')
print(dip.MSE(f,output3))
# dip.figure(1)
# dip.imshow(g,'gray')
# dip.figure(2)
# dip.imshow(f,'gray')
# dip.figure(3)
# dip.imshow(output1,'gray')
# dip.title('Gaussian Filter')
# dip.show()
