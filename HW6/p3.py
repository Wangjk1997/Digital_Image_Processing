import numpy as np
import dippykit as dip
import matplotlib.pyplot as plt

def KLT_blocks(X: np.ndarray,k,B) -> np.ndarray:
    row, col = X.shape

    #checking divisiblity and padding
    row_r = row % B
    col_r = col % B
    if (row_r != 0) or (col_r != 0):
        X = np.pad(X,((0,row_r),(0,col_r)),'constant',constant_values=(0,0))

    #new row, col and number of blocks in row, number of blocks in column
    row_new, col_new = X.shape
    row_block = int(row_new/B)
    col_block = int(col_new/B)

    #Let F be the matrix of vectorized patch
    F = np.zeros((B * B, row_block * col_block))
    index = 0
    for i in range(row_block):
        for j in range(col_block):
            block = X[i * B: (i+1) * B, j * B: (j+1) * B]
            F[:, index:index+1] = block.reshape(B * B, 1, order='F')
            index = index+1

    #Compute the mean vector of F
    u = np.zeros((B * B, 1))  # You need to calculate u
    for i in range(row_block * col_block):
        tmp = F[:, i]
        u = np.add(u, tmp.reshape((B * B, 1)))
    u = 1 / (row_block * col_block) * u
    rowOnes = np.ones((1, row_block * col_block))
    U = np.dot(u, rowOnes)
    Y = F - U

    #Compute Rff
    Rff = 1/(row_block * col_block) * np.dot(Y, np.transpose(Y))

    #decompose Rff into the product of eigenvectors and eigen matrix
    V, L, V2 = np.linalg.svd(Rff)

    #approximate eigenvectors
    V_app = V[:,0:k]

    #The approximate of F
    F_app = np.dot(np.dot(V_app, np.transpose(V_app)), Y) + U

    #Reconstruct the image and return it
    X_rec = np.zeros((row_new, col_new))
    index = 0
    for i in range(row_block):
        for j in range(col_block):
            tmp = F_app[:, index]
            X_rec[i * B: (i+1) * B, j * B: (j+1) * B] = tmp.reshape(B, B, order='F')
            index = index + 1

    X_rec = X_rec[0:row, 0:col]
    return X_rec
#X = dip.im_read('airplane_downsample_gray_square.jpg')
X = dip.im_read('brussels_gray_square.jpg')

X_hat_1 = KLT_blocks(X,1,10)
X_hat_12 = KLT_blocks(X,12,10)
X_hat_24 = KLT_blocks(X,24,10)
diff_1 = X - X_hat_1
diff_12 = X - X_hat_12
diff_24 = X - X_hat_24

dip.figure(1)
dip.subplot(2,2,1)
dip.imshow(X,'gray')
dip.title('Original')
dip.subplot(2,2,2)
dip.imshow(X_hat_1,'gray')
dip.title('k = 1')
dip.subplot(2,2,3)
dip.imshow(X_hat_12,'gray')
dip.title('k = 12')
dip.subplot(2,2,4)
dip.imshow(X_hat_24,'gray')
dip.title('k = 24')

dip.figure(2)
dip.subplot(1,3,1)
dip.imshow(diff_1,'gray')
dip.colorbar()
dip.title('k = 1')
dip.subplot(1,3,2)
dip.imshow(diff_12,'gray')
dip.colorbar()
dip.title('k = 12')
dip.subplot(1,3,3)
dip.imshow(diff_24,'gray')
dip.colorbar()
dip.title('k = 24')

plt.figure(3)
x = []
y = []
M, N = X.shape
for i in range(64):
    print(i)
    x.append(i)
    X_hat = KLT_blocks(X,i+1,10)
    MSE = dip.MSE(X,X_hat)
    y.append(MSE)
plt.plot(x,y)
plt.xlabel('k')
plt.ylabel('MSE')
plt.show()
dip.show()