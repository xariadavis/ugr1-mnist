import numpy
import numpy as np
from art.estimators.classification import PyTorchClassifier

import data_processing.make_dataset as make_data
import data_processing.adversarial as adv
from models.train_model import train_model
from models.summaries.LeNet5 import LeNet
from models.predict_model import evaluate_model
from visualization.visualize import *
import torch.nn.functional as f
from torch.utils.data import TensorDataset, DataLoader
import torchvision.transforms as transforms
from tabulate import tabulate
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim

model = LeNet()
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)


def add_stats(table, model, dataset, accuracy, trained, adv_data):
    entry = [model, dataset, str(accuracy), str(trained), str(adv_data)]
    table.append(entry)
    headers = ["MODEL", "DATASET", "ACCURACY", "TRAINED (CLEAN)", "TRAINED (ADV)"]
    table_str = tabulate(table, headers, tablefmt="pipe")

    return table_str


def main():

    # If GPU is available use it
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # Define datasets
    train_set = make_data.training_set
    test_set = make_data.test_set

    # Define DataLoaders
    train_loader = make_data.train_loader
    test_loader = make_data.test_loader

    # Train the defined model using the LeNet5 architecture
    # train_model(num_epochs=5, model=model, optimizer=optimizer, criterion=criterion,
    #               train_loader=train_loader, device=device)

    # Save the model to a file (optional if there is already one saved)
    # FILE = '../model/model.pth'
    # torch.save(model.state_dict(), FILE)

    # Load the model from the file
    # model = LeNet()
    model.load_state_dict(torch.load('../model/model.pth', map_location=torch.device('cpu')))

    # Evaluate the accuracy of the model on the training dataset
    accuracy, predictions_array = evaluate_model(model=model, loader=train_loader)

    stats = []
    add_stats(stats, 'LeNet', 'Train', accuracy, True, False)

    # Get first 40 images from train_loader
    train_ex_images = []
    train_ex_labels = []

    # Add the batches to lists to make it easier to work with
    train_images = []
    train_labels = []

    # Define the range for the loop
    total_iterations = len(train_loader)

    # Create the progress bar
    print()
    progress_bar = tqdm(total=total_iterations, desc='Unpacking train_loader', unit='iteration')

    # Append batches to lists
    for batch_images, batch_labels in train_loader:
        train_images.append(batch_images)
        train_labels.append(batch_labels)
        progress_bar.update(1)
    progress_bar.close()
    print()

    # Convert lists to tensors
    train_data = torch.cat(train_images, dim=0)
    train_labels = torch.cat(train_labels, dim=0)

    # Plot the first 40 images from train_set
    plot_images(train_data[0:40].squeeze().numpy(), train_labels[0:40].squeeze().numpy(), predictions_array[0:40],
                'train_examples')

    # accuracy, predictions_array = evaluate_model(model=model, loader=test_loader)
    # add_stats(stats, 'Base', 'Test', accuracy, '-', 'On clean dataset')

    # Plot the first 40 images from test_set
    # num_images = range(0, 40)
    # plot_images(test_set.data[num_images].numpy(), test_set.targets[num_images].numpy(),
    #             predictions_array[num_images], 'test_examples')

    # Define epsilons that will be used to generate adversarial examples
    epsilons = [
        0.0,
        0.1,
        0.2,
        0.3,
        0.4,
        0.5,
        1.0,
    ]

    # Define model wrapper from ART
    classifier = PyTorchClassifier(model=model, input_shape=(1, 32, 32), nb_classes=10, loss=criterion,
                                   optimizer=optimizer)

    # Generate adversarial data
    adversarial_examples, original_labels, adversarial_labels = adv.generate_adversarial(classifier=classifier,
                                                                                         epsilons=epsilons,
                                                                                         loader=train_loader)

    # Refactor image data
    adversarial_examples = f.interpolate(adversarial_examples, size=(32, 32), mode='bilinear', align_corners=False)

    # Append adversarial examples to clean dataset
    # New total is 82,848 (Previously 60,000)
    train_data = torch.tensor(np.append(train_data, adversarial_examples, axis=0))
    train_labels = torch.tensor(np.append(train_labels, original_labels, axis=0))
    predictions_array = np.append(predictions_array, adversarial_labels, axis=0)

    # Create a dataset using the modified data
    modified_dataset = TensorDataset(train_data, train_labels)

    # Create a dataloader for the modified dataset
    modified_loader = DataLoader(modified_dataset, batch_size=64, shuffle=False)

    # Evaluate model (trained on cleaned data) on modified data
    accuracy, p = evaluate_model(model=model, loader=modified_loader)
    add_stats(stats, 'LeNet', 'Modified', accuracy, True, False)

    # This range of images was generated with an epsilon of 1.0
    adv_range = range(82807, 82847)
    plot_images(train_data[adv_range].squeeze().numpy(), train_labels[adv_range].squeeze().numpy(),
                predictions_array[adv_range], 'adv_pre_train')

    # Train the wrapped model on adversarial examples
    print('Training classifier...\n')
    classifier.fit(x=train_data.numpy(), y=train_labels.numpy(), batch_size=64, nb_epochs=5)

    accuracy, predictions_array = evaluate_model(model=model, loader=modified_loader)
    table = add_stats(stats, 'LeNet', 'Modified', accuracy, True, True)
    plot_images(train_data[adv_range].squeeze().numpy(), train_labels[adv_range].squeeze().numpy(),
                predictions_array[adv_range], 'adv_post_train')

    print(table)


if __name__ == '__main__':
    main()
