import os

path = "./data/factin_patches"

files_training = [name for name in os.listdir(path) if os.path.isfile(os.path.join(path, name))]
for i, name in enumerate(sorted(files_training)):
    os.rename(os.path.join(path, name), os.path.join(path, f"{str(i)}"))
