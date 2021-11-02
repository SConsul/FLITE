import os

dataset_root_dir = '/Users/sarthak/Documents/Autumn 2020-21/CS330/Course Project/FLITE/data/orbit_benchmark_224/'
for subdir, dirs, files in os.walk(dataset_root_dir):
    for i, img in enumerate(sorted(files)):
        if i % 10 != 0:
            img_file = os.path.join(subdir, img)
            if os.path.isfile(img_file):
                os.remove(img_file)
            else:
                print("Error: %s file not found" % img_file)
