import numpy as np
from PIL import Image

sizeof_float = 4

def decompress_bmp(args):

    with open(args.in_file, 'rb') as f:
        compressed_image = f.read()

    width              = int(np.frombuffer(compressed_image, offset=0, dtype=np.uint32, count=1)[0])
    height             = int(np.frombuffer(compressed_image, offset=4, dtype=np.uint32, count=1)[0])
    compression_factor = int(np.frombuffer(compressed_image, offset=8, dtype=np.uint32, count=1)[0])

    offset  = 12
    U_size  = width  * compression_factor
    Vt_size = height * compression_factor
    
    URed       = np.frombuffer(compressed_image, offset=offset, dtype=np.float32, count=U_size).reshape((width, compression_factor))
    offset     += URed.size * sizeof_float
    SigmaRed   = np.diag(np.frombuffer(compressed_image, offset=offset, dtype=np.float32, count=compression_factor))
    offset     += compression_factor * sizeof_float
    VtRed      = np.frombuffer(compressed_image, offset=offset, dtype=np.float32, count=Vt_size).reshape((compression_factor, height))
    offset     += VtRed.size * sizeof_float

    UGreen     = np.frombuffer(compressed_image, offset=offset, dtype=np.float32, count=U_size).reshape((width, compression_factor))
    offset     += UGreen.size * sizeof_float
    SigmaGreen = np.diag(np.frombuffer(compressed_image, offset=offset, dtype=np.float32, count=compression_factor))
    offset     += compression_factor * sizeof_float
    VtGreen    = np.frombuffer(compressed_image, offset=offset, dtype=np.float32, count=Vt_size).reshape((compression_factor, height))
    offset     += VtGreen.size * sizeof_float

    UBlue      = np.frombuffer(compressed_image, offset=offset, dtype=np.float32, count=U_size).reshape((width, compression_factor))
    offset     += UBlue.size * sizeof_float
    SigmaBlue  = np.diag(np.frombuffer(compressed_image, offset=offset, dtype=np.float32, count=compression_factor))
    offset     += compression_factor * sizeof_float
    VtBlue     = np.frombuffer(compressed_image, offset=offset, dtype=np.float32, count=Vt_size).reshape((compression_factor, height))
    offset     += VtBlue.size * sizeof_float

    cRed   = URed   @ SigmaRed   @ VtRed
    cGreen = UGreen @ SigmaGreen @ VtGreen
    cBlue  = UBlue  @ SigmaBlue  @ VtBlue

    decompressed_image = Image.fromarray(np.uint8(np.stack((cRed, cGreen, cBlue), axis=-1)))
    decompressed_image.save(args.out_file)