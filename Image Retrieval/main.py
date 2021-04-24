import argparse
import os
import sys
import time
import pickle

from PIL import Image

from crop import ImageCropper


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("image", type=open, action="store")
    parser.add_argument("-c", "--crop", default=True, action="store_true")


    args = parser.parse_args()

    if args.crop:
        cropper = ImageCropper()
        cropper.set_file(args.image.name)
        cropper.run()
        photo = cropper.outputname
        with open('bbx.pkl', 'wb') as f:
            pickle.dump(cropper.box, f)
    else:
        photo = args.image.name