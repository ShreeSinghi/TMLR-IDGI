import argparse
import os
from PIL import Image
from tqdm import tqdm
from PIL import Image, ImageFilter


def blur_images(input_dir, output_dir, blur_radius=20):
    os.makedirs(output_dir, exist_ok=True)

    image_files = [f for f in os.listdir(input_dir) if os.path.isfile(os.path.join(input_dir, f))]

    for image_file in tqdm(image_files):
        with Image.open(os.path.join(input_dir, image_file)) as img:
            blurred_img = img.filter(ImageFilter.GaussianBlur(radius=blur_radius))
            blurred_img.save(os.path.join(output_dir, image_file))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataroot', type=str, required=True, help='path to dataset')

    opt = parser.parse_args()

    input_directory = os.path.join(opt.dataroot, r"val/images")
    os.makedirs(os.path.join(opt.dataroot, r"val/blurred"), exist_ok=True)
    output_directory = os.path.join(opt.dataroot, os.path.join(opt.dataroot, r"val/blurred"))

    blur_images(input_directory, output_directory)
