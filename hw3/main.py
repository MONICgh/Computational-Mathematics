import argparse

from compress import compress_bmp
from decompress import decompress_bmp

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["compress", "decompress"], required=True)
    parser.add_argument("--method", choices=["numpy", "simple", "advanced"])
    parser.add_argument("--compression_factor", type=int)
    parser.add_argument("--in_file", required=True)
    parser.add_argument("--out_file", required=True)
    args = parser.parse_args()
    
    if args.mode == "compress": compress_bmp(args)
    else:                       decompress_bmp(args)