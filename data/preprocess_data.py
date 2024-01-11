import json

from PIL import Image


def resize_image(image, to_width, to_height):
    return image.resize((to_width, to_height), Image.Resampling.LANCZOS)


def format_dataset(raw_dataset, width=800, height=1200):
    dataset = raw_dataset
    for ind in range(len(dataset['train'])):
        print("Processing image " + str(ind) + " of " + str(len(dataset['train'])))

        annotations = json.loads(dataset['train'][ind]['ground_truth'])

        width_scale = width / annotations['meta']['image_size']['width']
        height_scale = height / annotations['meta']['image_size']['height']

        # Resize the images by the width and height scale
        resized_image = resize_image(dataset['train'][ind]['image'], width, height)
        dataset['train'][ind]['image'] = resized_image

        # Parse the annotations for all images
        lines_to_parse = annotations['valid_line']

        for line in lines_to_parse:
            for word in line['words']:
                quad = word['quad']

                print("Old: " + str(quad['x1']) + " " + str(quad['x2']) + " " + str(quad['x3']) + " " + str(quad['x4']) + " "
                      + str(quad['y1']) + " " + str(quad['y2']) + " " + str(quad['y3']) + " " + str(quad['y4']))

                quad['x1'] = int(quad['x1'] * width_scale)
                quad['x2'] = int(quad['x2'] * width_scale)
                quad['x3'] = int(quad['x3'] * width_scale)
                quad['x4'] = int(quad['x4'] * width_scale)
                quad['y1'] = int(quad['y1'] * height_scale)
                quad['y2'] = int(quad['y2'] * height_scale)
                quad['y3'] = int(quad['y3'] * height_scale)
                quad['y4'] = int(quad['y4'] * height_scale)

                print("New: " + str(quad['x1']) + " " + str(quad['x2']) + " " + str(quad['x3']) + " " + str(quad['x4']) + " "
                      + str(quad['y1']) + " " + str(quad['y2']) + " " + str(quad['y3']) + " " + str(quad['y4']))

    return dataset
