import os
import shutil
import json
import argparse
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument('--dataroot', type=str, required=True, help='path to dataset')
parser.add_argument('--model', type=str, required=True, help='model name')
opt = parser.parse_args()

# Replace with the actual paths and dictionary of filename to class index
with open(os.path.join(opt.dataroot, f'{opt.model}_predictions.json'), "r") as file:
    image_class_dict = json.loads(file.read())

class_image_dict = {i: [] for i in range(1000)}

for image, classs in image_class_dict.items():
    class_image_dict[classs].append(image)

json.dumps()
# destination_folder = os.path.join(opt.dataroot, opt.model)

# os.makedirs(destination_folder, exist_ok=True)

# # Organize images by class
# for filename, class_index in tqdm(image_class_dict.items()):
#     class_folder = os.path.join(destination_folder, str(class_index))

#     # Create class folder if it doesn't exist
#     os.makedirs(class_folder, exist_ok=True)

#     # Source and destination paths for image copying
#     source_path = os.path.join(opt.dataroot, "val\\images", filename)
#     destination_path = os.path.join(class_folder, filename.split("\\")[-1])

#     # Copy the image to its corresponding class folder
#     shutil.copy(source_path, destination_path)