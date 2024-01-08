from datasets import load_dataset
# Load the dataset
dataset = load_dataset("naver-clova-ix/cord-v2")
from PIL import Image
import matplotlib.pyplot as plt
import pandas as pd
import json
from tqdm.auto import tqdm
# Splitting the dataset into training and test sets
from sklearn.model_selection import train_test_split


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


# Function to parse the ground truth JSON data and handle inconsistencies
def parse_annotations(annotations):
    # Load the JSON data
    gt_data = json.loads(annotations['ground_truth'])
    # Extract the 'menu' data
    menu_data = gt_data['gt_parse'].get('menu', [])
    # Ensure menu_data is a list
    if not isinstance(menu_data, list):
        menu_data = [menu_data]
    # Convert to DataFrame
    return pd.DataFrame(menu_data)

# # Parse the annotations for all images
# annotations_df_list = []
# for i in tqdm(range(5)):
#     annotations_df_list.append(parse_annotations(dataset['train'][i]))

for annotation in tqdm(dataset['train']):
    annotations_df_list.append(parse_annotations(annotation))

# Combine all the DataFrames into a single DataFrame
annotations_df = pd.concat(annotations_df_list, ignore_index=True)

# Function to clean and convert price-related columns to numeric
def clean_price_column(df, column_name):
    # Remove commas and convert to numeric, coerce errors to NaN
    df[column_name] = pd.to_numeric(df[column_name].str.replace(',', ''), errors='coerce')
    return df

# Clean the 'price' column
annotations_df = clean_price_column(annotations_df, 'price')

# Handling missing values
# Dropping columns with a high percentage of missing values
threshold = 0.5 # Columns with more than 50% missing values
annotations_df = annotations_df.dropna(thresh=len(annotations_df) * threshold, axis=1)

# For the remaining columns with missing values, we can impute
# For numerical columns, we'll use the median
for col in annotations_df.select_dtypes(include='number').columns:
    annotations_df[col].fillna(annotations_df[col].median(), inplace=True)

# For categorical columns, we'll use the mode
for col in annotations_df.select_dtypes(include='object').columns:
    annotations_df[col].fillna(annotations_df[col].mode()[0], inplace=True)

