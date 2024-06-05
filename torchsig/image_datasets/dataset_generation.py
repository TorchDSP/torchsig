import os
import cv2
import time
from torch import stack as stack

"""
saves a list of images and a list of labels as a png image file and a txt label file in yolo format
Inputs:
    images: list of images to be saved as a list of torch tensors
    labels: list of labels to be saved as a list of torch tensors or tuples of the form (class_id, center_x, center_y, width, height)
    path: a string filepath to the root directory for the dataset; it will contain subdirectories at <path>/images and <path>/labels; if these folders do not exist they will be created
    file_prefix: a string to prepend to the name of all generated files; used for batching or formatting file names
    black_hot: whether the output images are black-hot [if true, signals will appear black against a white background in the image files]
"""
def save_yolo_data(images, labels, path="./", file_prefix="1_", black_hot = True):
    images_path = path + "images/"
    labels_path = path + "labels/"
    images = stack(images)*255
    if black_hot:
        images = 255 - images
    np_images = images.cpu().numpy().transpose(0,2,3,1)
    for i in range(len(images)):
        image_fname = images_path+file_prefix+str(i)+".png"
        labels_fname = labels_path+file_prefix+str(i)+".txt"
        cv2.imwrite(image_fname, np_images[i])
        with open(labels_fname,'w') as labels_file:
            for label in labels[i]:
                labels_file.write(str(label[0])+" "+str(label[1])+" "+str(label[2])+" "+str(label[3])+" "+str(label[4])+"\n")

"""
batch-by-batch generates and saves in yolo format a dataset of specified size from a torch Dataset object (this will only work on synthetic datasets which output in yolo format)
Inputs:
    dataset: the source dataset to use for generation
    dataset_size: the desired size of the saved dataset
    output_path: the desired root directory of the saved dataset; it will contain subdirectories at <output_path>/images and <output_path>/labels; if these folders do not exist they will be created
    batch_size: the number of images to generate at once before saving them; this will be useful if generation is interrupted
    black_hot: whether the output images are black-hot [if true, signals will appear black against a white background in the image files]
    verbose: whether or not to print progress updates and total time taken to console
    batch_num: the number of the last batch completed before this function was called; used to restart generation of interrupted; defaults to -1, which will generate the whole dataset
"""
def batched_write_yolo_synthetic_dataset(dataset, dataset_size, output_path, batch_size=1000, verbose=False, black_hot=True, batch_num=-1):
    if verbose:
        stime = time.time()
    images_path = output_path + "images/"
    labels_path = output_path + "labels/"
    os.makedirs(output_path, exist_ok=True)
    os.makedirs(images_path, exist_ok=True)
    os.makedirs(labels_path, exist_ok=True)
    
    num_generated = (batch_num + 1)*batch_size
    
    while num_generated + batch_size < dataset_size:
        batch_num += 1
        num_generated += batch_size
        images = []
        labels = []
        for i in range(batch_size):
            image, label_set = dataset[0]
            images += [image]
            labels += [label_set]

        save_yolo_data(images, labels, output_path, str(batch_num)+"_", black_hot=black_hot)
        if verbose:
            print("...batch #"+str(batch_num)+" complete...")
    batch_num += 1
    images = []
    labels = []
    if dataset_size % batch_size > 0:
        for i in range(dataset_size % batch_size):
            image, label_set = dataset[0]
            images += [image]
            labels += [label_set]
        save_yolo_data(images, labels, output_path, str(batch_num)+"_", black_hot=black_hot)
    if verbose:
        print("...done!")
        etime = time.time()
        print("total time: ",str(etime-stime),"seconds")