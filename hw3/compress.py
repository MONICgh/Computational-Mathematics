import numpy as np
from PIL import Image


def simple(matrix):
    if matrix.shape[0] >= matrix.shape[1]: matrix_sq = np.dot(matrix, matrix.T)
    else:                                  matrix_sq = np.dot(matrix.T, matrix)
    
    eig_val, eig_vect = np.linalg.eig(matrix_sq)
    sort_val = np.argsort(eig_val)[::-1]
    eig_val  = eig_val[sort_val]
    eig_vect = eig_vect[:, sort_val]

    return (
        np.dot(matrix, eig_vect) / np.linalg.norm(np.dot(matrix, eig_vect), axis=0),
        np.sqrt(eig_val),
        eig_vect.T
    )

def advanced(matrix, k):
    height, width = matrix.shape

    vectors_rand = []
    for _ in range(width):
        rand = np.random.normal(size=k) 
        vectors_rand.append(rand / np.linalg.norm(rand))

    d_change, flag_break = 1, 0
    while d_change > 0.01 and flag_break < 15:
        
        q, _ = np.linalg.qr(matrix @ vectors_rand)
        u    = np.matrix(q[:, 0:k])

        q, r         = np.linalg.qr(matrix.T @ u)
        vectors_rand = np.matrix(q[:, 0:k])
        sigma        = np.matrix(r[0:k, 0:k])

        d_change = np.linalg.norm(matrix @ vectors_rand - u * sigma)
        flag_break += 1

    return (u, np.diagonal(sigma).astype(np.float32), vectors_rand.T)

def compress_bmp(args):
    
    imgage = Image.open(args.in_file)
    height, width = imgage.size

    compression_factor = int(height * width / (8 * args.compression_factor * (height + width + 1)))
    
    cRed   = np.array(imgage, dtype='float64')[:,:,0]
    cGreen = np.array(imgage, dtype='float64')[:,:,1]
    cBlue  = np.array(imgage, dtype='float64')[:,:,2]

    if args.method == "numpy":
        URed,   SigmaRed,   VtRed   = np.linalg.svd(cRed,   full_matrices=False)
        UGreen, SigmaGreen, VtGreen = np.linalg.svd(cGreen, full_matrices=False)
        UBlue,  SigmaBlue,  VtBlue  = np.linalg.svd(cBlue,  full_matrices=False)
    elif args.method == "simple":
        URed,   SigmaRed,   VtRed   = simple(cRed)
        UGreen, SigmaGreen, VtGreen = simple(cGreen)
        UBlue,  SigmaBlue,  VtBlue  = simple(cBlue)
    else:
        URed,   SigmaRed,   VtRed   = advanced(cRed, compression_factor)
        UGreen, SigmaGreen, VtGreen = advanced(cGreen, compression_factor)
        UBlue,  SigmaBlue,  VtBlue  = advanced(cBlue, compression_factor)

    URed   = URed[:,:compression_factor]
    UGreen = UGreen[:,:compression_factor]
    UBlue  = UBlue[:,:compression_factor] 

    SigmaRed   = SigmaRed[:compression_factor]
    SigmaGreen = SigmaGreen[:compression_factor]
    SigmaBlue  = SigmaBlue[:compression_factor]

    VtRed   =VtRed[:compression_factor,:]
    VtGreen =VtGreen[:compression_factor,:]
    VtBlue  =VtBlue[:compression_factor,:]

    
    compressed_image = bytearray()
    compressed_image.extend(np.uint32(width).tobytes())
    compressed_image.extend(np.uint32(height).tobytes())
    compressed_image.extend(np.uint32(compression_factor).tobytes())

    compressed_image.extend(URed.astype(dtype=np.float32).tobytes())
    compressed_image.extend(SigmaRed.astype(dtype=np.float32).tobytes())
    compressed_image.extend(VtRed.astype(dtype=np.float32).tobytes())

    compressed_image.extend(UGreen.astype(dtype=np.float32).tobytes())
    compressed_image.extend(SigmaGreen.astype(dtype=np.float32).tobytes())
    compressed_image.extend(VtGreen.astype(dtype=np.float32).tobytes())

    compressed_image.extend(UBlue.astype(dtype=np.float32).tobytes())
    compressed_image.extend(SigmaBlue.astype(dtype=np.float32).tobytes())
    compressed_image.extend(VtBlue.astype(dtype=np.float32).tobytes())

    with open(args.out_file, 'wb') as f:
        f.write(compressed_image)
