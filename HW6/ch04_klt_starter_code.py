import dippykit as dip
import numpy as np

# Step 1: Defining the image
X = np.array([[1, 1, 1, 1, 1],
              [1, 1, 2, 2, 2],
              [0, 2, 1, 0, 0],
              [0, 2, 1, 0, 1]])


M, N = X.shape  # Reading the dimensions of the image
# Step 2: Calculating the mean vector
# ============================ EDIT THIS PART =================================
u = np.zeros((M, 1))  # You need to calculate u
for i in range (N):
    tmp = X[:,i]
    u = np.add(u,tmp.reshape((M,1)))
u = 1/N*u

# Step 3: Subtracting the mean
# ============================ EDIT THIS PART =================================
Y = np.zeros((M, N))  # You need to calculate Y
rowOnes = np.ones((1, N))
U = np.dot(u, rowOnes)
Y = X - U
# Step 4: Calculating the autocorrelation/covariance matrix
# ============================ EDIT THIS PART =================================
Ry = np.zeros((M, M))  # You need to calculate Ry
Ry = 1/N * np.dot(Y, np.transpose(Y))
# Step 5: Finding the eigenvectors
V = np.zeros((M, N))
V, L, V2 = np.linalg.svd(Ry)
# ============================ EDIT THIS PART =================================

# Calculate the eigenvectors and put them in as columns of the matrix V.
# You may use the function "np.linalg.svd" which is the same as
# "np.linalg.eig" for positive semi-definite matrices but it orders the
# vectors in a descending order of the eigenvalues




# Step 6: Define the transformation matrix
# ============================ EDIT THIS PART =================================
A = np.zeros((M, M))  # Define A
A = np.transpose(V)
# STEP 7: Calculating the KLT
# ============================ EDIT THIS PART =================================
Z = np.zeros((M, N))  # Calculate the KLT of the X
Z = np.dot(A,Y)
print('Z')
print(Z)
# STEP 7: Verification of results:
# ============================ EDIT THIS PART =================================
Rz = np.zeros((M, M))
Rz = 1 / N * np.dot(Z,np.transpose(Z))
print('Rz')
print(Rz)
# Step 8: Inverse Transform:
# ============================ EDIT THIS PART =================================
X_rec = np.zeros((M, N))
X_rec = np.dot(np.transpose(A),Z) + U
print('X_rec')
print(X_rec)

# ===================!!!!! DO NOT EDIT THIS PART !!!!!=========================
ReconstructionError = np.linalg.norm(X - X_rec, ord='fro')
print('Reconstruction Error:', ReconstructionError)

