from datasets import load_dataset
import resize_data
import matplotlib.pyplot as plt

# Load the dataset
dataset = load_dataset("naver-clova-ix/cord-v2")
# print(dataset)

# # Access the first example in the 'train' split
# train_example = dataset['train'][0]

# # Access specific values in the example
# train_image_value = train_example["image"]
# train_ground_truth_value = train_example["ground_truth"]

# Print the original values
print(f"Original Ground Truth: {train_ground_truth_value}")

# Call resize_data function
resized_image = resize_data.resize_image(train_image_value, 800)

# Display the resized image
plt.figure(figsize=(5, 5))
plt.imshow(resized_image)
plt.title('Resized Image')
plt.axis('off')
plt.show()

