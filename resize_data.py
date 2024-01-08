from datasets import load_dataset
# Load the dataset
dataset = load_dataset("naver-clova-ix/cord-v2")
from PIL import Image
import matplotlib.pyplot as plt

# Function to resize images while maintaining aspect ratio
def resize_image(image, base_width):
    w_percent = (base_width / float(image.size[0]))
    h_size = int((float(image.size[1]) * float(w_percent)))
    return image.resize((base_width, h_size), Image.Resampling.LANCZOS)

# Resize the first few images to a width of 800 pixels
resized_images = []
for i in range(2):
    image = dataset['train'][i]['image']
    resized_image = resize_image(image, 800)
    resized_images.append(resized_image)

# Display the resized images
for i, img in enumerate(resized_images):
    plt.figure(figsize=(5,5))
    plt.imshow(img)
    plt.title('Resized Image ' + str(i+1))
    plt.axis('off')
    plt.show()