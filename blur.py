from __future__ import print_function
import argparse
import os
import numpy as np
import json
from matplotlib import pyplot as plt
from PIL import Image
import gc
from tqdm import tqdm
from PIL import Image, ImageFilter


def blur_images(input_dir, output_dir, blur_radius=20):
    # Create the output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Get a list of all files in the input directory
    image_files = [f for f in os.listdir(input_dir) if os.path.isfile(os.path.join(input_dir, f))]

    # Iterate over each image file
    for image_file in tqdm(image_files):
        # Open the image using PIL
        with Image.open(os.path.join(input_dir, image_file)) as img:
            # Apply Gaussian blur with the specified radius
            blurred_img = img.filter(ImageFilter.GaussianBlur(radius=blur_radius))

            # Save the blurred image to the output directory
            blurred_img.save(os.path.join(output_dir, image_file))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataroot', type=str, required=True, help='path to dataset')

    opt = parser.parse_args()

    # Specify the input and output directories
    input_directory = os.path.join(opt.dataroot, r"val/images")
    os.makedirs(os.path.join(opt.dataroot, r"val/blurred"), exist_ok=True)
    output_directory = os.path.join(opt.dataroot, os.path.join(opt.dataroot, r"val/blurred"))

    # Blur images in the input directory and save them to the output directory
    blur_images(input_directory, output_directory)
