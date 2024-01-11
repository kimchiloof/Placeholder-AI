import data.get_data as data
import matplotlib.pyplot as plt
import matplotlib.patches as patches


# Function to overlay annotations on an image
def overlay_annotations(_image, _annotations):
    # Prepare the plot
    fig, ax = plt.subplots(1)
    ax.imshow(_image)

    # Plot each annotation
    for annotation in _annotations['valid_line']:
        for word in annotation['words']:
            quad = word['quad']
            # Draw the rectangle around each word
            rect = patches.Rectangle((quad['x1'], quad['y1']), quad['x2'] - quad['x1'], quad['y3'] - quad['y1'],
                                     linewidth=1, edgecolor='r', facecolor='none')
            ax.add_patch(rect)

            # Annotate the text
            plt.text(quad['x1'], quad['y1'], word['text'], color='white', fontsize=8)

    plt.show()


for i in range(len(data.dataset['train'])):
    image = data.dataset['train'][i]['image']
    annotations = data.dataset['train'][i]['ground_truth']
    overlay_annotations(image, annotations)
