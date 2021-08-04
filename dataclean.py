import shutil
import os
import random
from os import listdir

from os.path import isfile, join


source = r"image\oregon_wildlife\oregon_wildlife"
training_destination = r"image\train\nm"
val_destination = r"image\val\nm"
sep = "\\"
print(sep)

image_folder_name = [f for f in listdir(source)]

for name in image_folder_name:
    file_name_list = [f for f in listdir(source + sep + name) if not f[-4:] == ".gif"]
    traning_list = random.sample(file_name_list, 5)

    for i in traning_list:
        shutil.move(source + sep + name + sep + i, training_destination + sep + name + i)

    file_name_list = [f for f in listdir(source + sep + name) if not f[-4:] == ".gif"]
    val_list = random.sample(file_name_list, 2)
    for i in val_list:
        shutil.move(source + sep + name + sep + i, val_destination + sep + name + i)
