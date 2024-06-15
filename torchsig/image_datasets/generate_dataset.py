import os
from torchsig.image_datasets.datasets.synthetic_signals import GeneratorFunctionDataset, rectangle_signal_generator_function, tone_generator_function, repeated_signal_generator_function
from torchsig.image_datasets.datasets.file_loading_datasets import SOIExtractorDataset, LazyImageDirectoryDataset
from torchsig.image_datasets.datasets.composites import YOLOImageCompositeDataset, CombineDataset
from torchsig.image_datasets.transforms.impairments import BlurTransform, RandomGaussianNoiseTransform, RandomImageResizeTransform, RandomRippleNoiseTransform, ScaleTransform, scale_dynamic_range, normalize_image
from torchsig.image_datasets.dataset_generation import batched_write_yolo_synthetic_dataset

"""
constants/config stuff---------------------------------------------------------------------------------------------------------
"""
TRAINING_PATH = "./new_dataset_sig53_imgs/training/"
TESTING_PATH = "./new_dataset_sig53_imgs/testing/"
SOI_FOLDER_PATH = "<PATH_TO_SPECTROGRAM_IMAGES_MARKED_WITH_SIGNAL_OF_INTEREST>"
signal_image_folder_path = "<PATH_TO_SPECTROGRAM_IMAGES>"

NUM_TRAINING_DATA = 250000
NUM_TESTING_DATA = 25000

spectrogram_size = (1,1024,1024)
"""
--------------------------------------------------------------------------------------------------------------------------------
"""

modulations_paths = [signal_image_folder_path + fpath + "/" for fpath in os.listdir(signal_image_folder_path)]
image_datasets = []
for path in modulations_paths:
    image_datasets += [LazyImageDirectoryDataset(path, 0, read_black_hot=True, transforms = [normalize_image, lambda x: x**30, normalize_image, lambda x: x[0,(x.sum(axis=2)>1)[0]][None,:,:]**(1/10), normalize_image, scale_dynamic_range])]



ripple_transform = RandomRippleNoiseTransform((0.1,0.3), num_emitors=8, base_freq=3)

tone_transforms = []
tone_transforms += [BlurTransform(strength=0.3, blur_shape=(2,1))]
tone_transforms += [ripple_transform]
tone_dataset = GeneratorFunctionDataset(tone_generator_function(spectrogram_size[-1], max_height=40), 0, transforms=tone_transforms)

signal_transforms = [normalize_image]
signal_transforms += [RandomImageResizeTransform([0.6,3])]
signal_transforms += [BlurTransform(strength=1, blur_shape=(5,1))]
signal_transforms += [ripple_transform]
image_ds = CombineDataset(image_datasets, transforms = signal_transforms)


soi_transforms = [normalize_image]
soi_transforms += [RandomImageResizeTransform([0.6,1.5])]
soi_transforms += [BlurTransform(strength=1, blur_shape=(2,1))]
soi_transforms += [ripple_transform]
soi_transforms += [scale_dynamic_range]
#soi_dataset; path is the path to a folder of images with sois in drawn on bounding boxes; second argument is class label (here 1, vs 0 in other data)
#--- read_black_hot determines if it should assume the soi images are whire hot or black hot; filter strength is how aggressively to filter out background in he soi extraction
soi_dataset = SOIExtractorDataset(SOI_FOLDER_PATH,1, transforms=signal_transforms,read_black_hot=True, filter_strength=20)



repeat_image_ds = CombineDataset(image_datasets, transforms = [RandomImageResizeTransform([0.6,3])])
repeater_transforms = [BlurTransform(strength=1, blur_shape=(5,1)), ripple_transform]
repeater_signal_dataset = GeneratorFunctionDataset(repeated_signal_generator_function(lambda: repeat_image_ds[0][0], min_gap=10, max_gap=40, min_repeats=3, max_repeats=6), 0, transforms=repeater_transforms)

#these transforms, the signal_transforms, and the soi_transforms are the important ones that significantly change how things look; if something doesn't look right the problem is usually here
composite_transforms = []
composite_transforms += [BlurTransform(strength=0.2, blur_shape=(20,2))] #helps simulate rolloff, nd helps things look seemless in the composite image
composite_transforms += [normalize_image] # inf norm
composite_transforms += [RandomRippleNoiseTransform((0.3,0.5), num_emitors=8, base_freq=3)] #light but noticeable ripple in the full image;
composite_transforms += [RandomGaussianNoiseTransform(mean=0, range=(0.2,0.4))]
composite_transforms += [scale_dynamic_range] #adjusts the noise floor in each vertical column
composite_transforms += [normalize_image] # inf norm

#set the min_to_add and max_to_add to decide how many of a thing will appear in each image generated
composite_spectrogram_dataset = YOLOImageCompositeDataset(spectrogram_size, transforms=composite_transforms, dataset_size=NUM_TRAINING_DATA)#dataset_size is mostly obsolete, but still needs to be set, for reasons;
composite_spectrogram_dataset.add_component(tone_dataset, min_to_add=0, max_to_add=2)
composite_spectrogram_dataset.add_component(image_ds, min_to_add=0, max_to_add=3)
composite_spectrogram_dataset.add_component(soi_dataset, min_to_add=1, max_to_add=2)
composite_spectrogram_dataset.add_component(repeater_signal_dataset, min_to_add=0, max_to_add=2)

#batch_size=100 seems to work best, but it doesn't make a lot of difference to change it; might work different on a different machine
print("...making training dataset")
batched_write_yolo_synthetic_dataset(composite_spectrogram_dataset, NUM_TRAINING_DATA, TRAINING_PATH, verbose=True, black_hot=True, batch_size = 100)

print("...making testing dataset")
batched_write_yolo_synthetic_dataset(composite_spectrogram_dataset, NUM_TESTING_DATA, TESTING_PATH, verbose=True, black_hot=True, batch_size = 100)