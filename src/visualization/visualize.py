import matplotlib.pyplot as plt
import math
import numpy as np


def format_plot(num_images):
    if num_images == 1:
        rows, columns = 1, 1  # Set rows and columns to 1 for a single image
    else:
        sqrt = int(math.sqrt(num_images))
        factors = [(i, num_images // i) for i in range(2, sqrt + 1) if num_images % i == 0]

        if not factors:
            columns, rows = 1, num_images  # Set rows to 1 and columns to num_images
        else:
            columns, rows = factors[-1]  # Get the last (largest) factor pair

    fig, ax = plt.subplots(nrows=rows, ncols=columns, figsize=(columns * 3, rows * 3))
    ax = np.ravel(ax)  # Flatten the Axes array

    return fig, ax


def plot_images(images, ground_truths, predicted_digits, title):
    fig, ax = format_plot(len(images))

    for i, input_image in enumerate(images):

        curr = ax.ravel()[i]
        curr.imshow(input_image, cmap='gray')

        text_color = "green" if predicted_digits[i] == ground_truths[i] else "red"
        curr.set_title(f'Predicted: {predicted_digits[i]}', color=text_color, fontsize=14)
        curr.set_xlabel(f'Ground Truth: {ground_truths[i]}', fontsize=12)
        curr.xaxis.set_label_position('top')
        curr.set_xticks([])
        curr.set_yticks([])

    plt.tight_layout()
    plt.savefig('../reports/figures/' + title + '.png')
    plt.show()


def plot_adversarial_accuracy(attack_name, epsilon, accuracy):
    # Plot accuracy by epsilon on a plot
    plt.title(attack_name)
    plt.xlabel('Epsilon')
    plt.ylabel('Accuracy(%)')
    plt.plot(epsilon, accuracy, marker='o')
    plt.savefig('../reports/figures/' + attack_name + '.png')
