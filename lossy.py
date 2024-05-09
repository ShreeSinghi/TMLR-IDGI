import argparse
import os
from PIL import Image
from tqdm import tqdm

def lossy_images(input_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    image_files = [f for f in os.listdir(input_dir) if os.path.isfile(os.path.join(input_dir, f))]

    for image_file in tqdm(image_files):
        with Image.open(os.path.join(input_dir, image_file)) as img:
            img.save(os.path.join(output_dir, image_file), format='JPEG', quality=75)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataroot', type=str, required=True, help='path to dataset')

    opt = parser.parse_args()

    input_directory = os.path.join(opt.dataroot, r"val/images")
    os.makedirs(os.path.join(opt.dataroot, r"val/lossy"), exist_ok=True)
    output_directory = os.path.join(opt.dataroot, os.path.join(opt.dataroot, r"val/lossy"))

    lossy_images(input_directory, output_directory)
