import os
import numpy as np
from jupyter_bbox_widget import BBoxWidget
import ipywidgets as widgets
from matplotlib import pyplot as plt

def setup_yolo_directories(root_path):
    image_dir = root_path + "images/"
    label_dir = root_path + "labels/"
    
    if not os.path.isdir(root_path):
        os.mkdir(root_path)
    if not os.path.isdir(image_dir):
        os.mkdir(image_dir)
    if not os.path.isdir(label_dir):
        os.mkdir(label_dir)

def load_and_process_image(fpath):
    img = plt.imread(fpath)
    return img
def save_image(img, fpath):
    plt.imsave(fpath, img)
def save_yolo_labels(labels, fpath):
    with open(fpath,'w') as labels_file:
            for label in labels:
                labels_file.write(str(label[0])+" "+str(label[1])+" "+str(label[2])+" "+str(label[3])+" "+str(label[4])+"\n")

def save_as_yolo_data(output_image_dir, output_label_dir, fname, img, bboxes, class_names):
    """
    Saves data from the annotator widget as yolo image/label files in the output directory
    Inputs:
        output_image_dir - the path of the image directory for the new yolo data
        output_label_dir - the path of the label directory for the new yolo data
        fname - the name of the image being saved
        img - the image being saved
        bboxes - the bounding boxes to be saved
    """
    height, width = img.shape[:2]
    labels = []
    for box in bboxes:
        cid = class_names.index(box['label'])
        cx = (box['x'] + box['width']//2)/width
        cy = (box['y'] + box['height']//2)/height
        new_width = box['width']/width
        new_height = box['height']/height
        labels += [[cid, cx, cy, new_width, new_height]]
    save_image(img, output_image_dir + fname)
    label_fname = fname[:-4] + ".txt"
    save_yolo_labels(labels, output_label_dir + label_fname)

def yolo_annotator(input_image_dir, output_root_path, class_names=['Signal']):
    """
    loads and runs an interactive notebook cell with an annotation tool that lets ou label the images in input_image_dir in yolo format and save the outputs to output_root_path
    annotations are saved as you label them, and the tool will recognize and skip images which already have labels, so terminating and reruning the tool will pick up labeling on the next unlabeled image
    by default the tool uses a single 'signal' class, but an array of string class_names can be passed in
    """
    setup_yolo_directories(output_root_path)
    fnames = os.listdir(input_image_dir)
    annotated_fnames = os.listdir(output_root_path + "images/") # used to make sure we don't annotate already annotated images
    fname_ind = 0
    fname = fnames[fname_ind]
    while fname in annotated_fnames:
        fname_ind += 1
        if fname_ind >= len(fnames):
            raise IndexError("There are no more unlabeled images is the target directory. Either remove existing label and image files from the output dataset, specify a new input directory to add new data to the dataset, or specify a new output directory to relabel images in a new dataset.")
        fname = fnames[fname_ind]
    annotation_tool = BBoxWidget(
        image = os.path.join(input_image_dir, fname),
        classes=class_names
    )
    annotation_tool.fnames = fnames
    annotation_tool.annotated_fnames = fnames
    annotation_tool.ind = fname_ind

    out_cell = widgets.Output(layout={'border': '1px solid black'})

    # when Skip button is pressed we move on to the next file
    @annotation_tool.on_skip
    def skip():
        annotation_tool.ind += 1
        if annotation_tool.ind >= len(annotation_tool.fnames):
            out_cell.append_display_data("There are no more unlabeled images is the target directory. Either remove existing label and image files from the output dataset, specify a new input directory to add new data to the dataset, or specify a new output directory to relabel images in a new dataset.")
            annotation_tool.close()
            print("All input images are labeled")
            return "All input images are labeled"
        annotation_tool.fname = annotation_tool.fnames[annotation_tool.ind]
        if not fname in annotated_fnames:
            annotation_tool.image = os.path.join(input_image_dir, annotation_tool.fname)
            annotation_tool.bboxes = [] 
        else:
            skip()
    
    # when Submit button is pressed we save current annotations
    # and then move on to the next file
    @annotation_tool.on_submit
    def submit():
        annotation_tool.fname = annotation_tool.fnames[annotation_tool.ind]
        img = load_and_process_image(input_image_dir + annotation_tool.fname)
        save_as_yolo_data(output_root_path + "images/", output_root_path + "labels/", annotation_tool.fname, img, annotation_tool.bboxes, annotation_tool.classes)
        skip()

    out_cell.append_display_data(annotation_tool)
    out_cell.annotation_tool = annotation_tool
    
    return out_cell

