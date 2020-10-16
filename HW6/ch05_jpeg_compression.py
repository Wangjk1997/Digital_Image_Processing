import dippykit as dip
import numpy as np
import cv2

# STEP 1: Loading the image
# ============================ EDIT THIS PART =================================
im = dip.im_read('lena.png')
im = dip.float_to_im(im, 8)

# STEP 2: Converting to YCrCb
# ============================ EDIT THIS PART =================================
im = dip.rgb2ycbcr(im)  # HINT: Look into dip.rgb2ycbcr

# STEP 3: Keep only the luminance channel
# ============================ EDIT THIS PART =================================
im = im[:,:,0]

# ===================!!!!! DO NOT EDIT THIS PART !!!!!=========================
dip.figure()
dip.subplot(2, 2, 1)
dip.imshow(im, 'gray')
dip.title('Grayscale image')
# STEP 4: Calculating the image entropy
# ============================ EDIT THIS PART =================================
H = dip.entropy(im)

# ===================!!!!! DO NOT EDIT THIS PART !!!!!=========================
print("Entropy of the grayscale image = {:.2f} bits/pixel".format(H))

# STEP 5: Coding of the original image
# ============================ EDIT THIS PART =================================
im_vec = im.reshape(-1)
im_encoded, stream_length, symbol_code_dict, symbol_prob_dict = dip.huffman_encode(im_vec)
row, col = im.shape
im_bit_rate = stream_length/(row * col)
# ===================!!!!! DO NOT EDIT THIS PART !!!!!=========================
print("Bit rate of the original image = {:.2f} bits/pixel"
      .format(im_bit_rate))

# STEP 6: Subtract 127
# ============================ EDIT THIS PART =================================
# Change im to a float for computations
im = im.astype(float)
im = im - 127

# STEP 7: Block-wise DCT
block_size = (8, 8)
# ============================ EDIT THIS PART =================================
im_DCT = dip.block_process(im,dip.dct_2d,block_size)

# ===================!!!!! DO NOT EDIT THIS PART !!!!!=========================
dip.subplot(2, 2, 3)
dip.imshow(im_DCT, 'gray')
dip.title("Block-wise DCT coefficients - Blocksize = {}x{}"
          .format(*block_size))

# STEP 8: Quantization
c = 1
Q_table = dip.JPEG_Q_table_luminance
# ============================ EDIT THIS PART =================================
quantize = lambda x: x/(c*Q_table)
im_DCT_quantized = dip.block_process(im_DCT,quantize,block_size)
im_DCT_quantized = im_DCT_quantized.astype(int)
# ===================!!!!! DO NOT EDIT THIS PART !!!!!=========================
dip.subplot(2, 2, 4)
dip.imshow(im_DCT_quantized, 'gray')
dip.title('Quantized DCT coefficients')

# STEP 9: Entropy Coding
q_bit_stream, q_bit_stream_length, q_symbol_code_dict, q_symbol_prob_dict= dip.huffman_encode(im_DCT_quantized.reshape(-1))
# ============================ EDIT THIS PART =================================
q_bit_rate = q_bit_stream_length/(row * col)

# ===================!!!!! DO NOT EDIT THIS PART !!!!!=========================
print("Bit rate of the compressed image = {:.2f} bits/pixel"
      .format(q_bit_rate))

# STEP 10: Saving the bitstream to a binary file
# ===================!!!!! DO NOT EDIT THIS PART !!!!!=========================
bit_stream_file = open("CompressedSunset.bin", "wb")
q_bit_stream.tofile(bit_stream_file)
bit_stream_file.close()

# STEP 11-i: Read the binary file
# ===================!!!!! DO NOT EDIT THIS PART !!!!!=========================
bit_stream_file = open("CompressedSunset.bin", "rb")
q_bit_stream = np.fromfile(bit_stream_file, dtype='uint8')
bit_stream_file.close()

# STEP 11-ii: Decoding
# ============================ EDIT THIS PART =================================
im_DCT_quantized_decoded = dip.huffman_decode(q_bit_stream,
        q_symbol_code_dict, init_arr_size=im.size)
im_DCT_quantized_decoded = im_DCT_quantized_decoded[0:row * col].reshape((row,col))

# ===================!!!!! DO NOT EDIT THIS PART !!!!!=========================
im_DCT_quantized_decoded = im_DCT_quantized_decoded[:im.size]
im_DCT_quantized_reconstructed = im_DCT_quantized_decoded.reshape(im.shape)

# STEP 12: Dequantization
# ============================ EDIT THIS PART =================================
dequantize = lambda x: x*(c*Q_table)
im_DCT_reconstructed = dip.block_process(im_DCT_quantized_reconstructed,dequantize,block_size)

# STEP 13: Inverse DCT
# ============================ EDIT THIS PART =================================
im_reconstructed = dip.block_process(im_DCT_reconstructed,dip.idct_2d,block_size)

# STEP 14: Add 127 to every pixel
# ============================ EDIT THIS PART =================================
im_reconstructed = im_reconstructed + 127

# ===================!!!!! DO NOT EDIT THIS PART !!!!!=========================
dip.subplot(2, 2, 2)
dip.imshow(im_reconstructed, 'gray')
dip.title('Reconstructed image')

# ===================!!!!! DO NOT EDIT THIS PART !!!!!=========================
im = im + 127

# STEP 15: Calculating MSE and PSNR
# ============================ EDIT THIS PART =================================
MSE = dip.MSE(im, im_reconstructed)
PSNR = dip.PSNR(im,im_reconstructed,np.max(im))

# ===================!!!!! DO NOT EDIT THIS PART !!!!!=========================
print("MSE = {:.2f}".format(MSE))
print("PSNR = {:.2f} dB".format(PSNR))

dip.show()

