import os
import json

def index_files(directory, json_file_path):
    if os.path.exists(json_file_path):
        with open(json_file_path, 'r') as json_file:
            file_index = json.load(json_file)
    else:
        file_index = {}

    for root, dirs, files in os.walk(directory):
        if 'Extracted Points' in dirs:
            dirs.remove('Extracted Points')  # Exclude the directory from further processing
            
        print(dirs)
        print()
        
        for file in files:
            print(file)
            file_path = os.path.join(root, file)
            if file_path not in list(file_index.values()):
                file_index[len(file_index) + 1] = file_path

    with open(json_file_path, 'w') as json_file:
        json.dump(file_index, json_file, indent=4)

directory_path = '.'

json_file_path = 'file_index.json'

index_files(directory_path, json_file_path)
