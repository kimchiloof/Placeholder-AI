from datasets import load_dataset
import tensorflow as tf

# Load the dataset
dataset = load_dataset("naver-clova-ix/cord-v2")
print(dataset)

# # Access the first example in the 'train' split
# train_example = dataset['train'][0]

# # Access specific values in the example
# train_image_value = train_example["image"]
# train_ground_truth_value = train_example["ground_truth"]

# # Print the values
# print(f"Train Example - Image: {train_image_value}, Ground Truth: {train_ground_truth_value}")
