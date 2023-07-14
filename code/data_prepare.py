import os
import random
import shutil

original_data_folder = 'crawl_pokemon'
new_data_folder = 'new_data'


folders = os.listdir(original_data_folder)

for task in range(600):
    selected_folders = random.sample(folders, 5)
    for index, folder in enumerate(selected_folders):
        new_folder_path = os.path.join(new_data_folder+'/'+str(task), str(index))

        shutil.copytree(os.path.join(original_data_folder, folder), new_folder_path)


