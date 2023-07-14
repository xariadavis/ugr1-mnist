from art.attacks.evasion import FastGradientMethod
from src.visualization.visualize import plot_adversarial_accuracy
import torch


# Generate adversarial examples
def generate_adversarial(classifier, epsilons, loader):

    adversarial_examples = []
    original_labels = []
    adversarial_labels = []
    count = 0
    correct = 0
    total = 0

    accuracy = []
    first_num = []

    for eps in epsilons:
        attack = FastGradientMethod(estimator=classifier, eps=eps)
        final_batch = count + 50
        # Generate adversarial examples for the dataset
        for images, labels in loader:
            images = images.float()
            labels = labels.long()

            # Generate adversarial examples
            adversarial = attack.generate(x=images.numpy())

            # Predict the labels for original and adversarial examples
            original_prediction = classifier.predict(images)
            adversarial_prediction = classifier.predict(adversarial)

            total += labels.size(0)
            correct += (torch.argmax(torch.from_numpy(original_prediction), dim=1) ==
                        torch.argmax(torch.from_numpy(adversarial_prediction), dim=1)).sum().item()

            adversarial_examples.append(adversarial)
            original_labels.append(original_prediction)
            adversarial_labels.append(adversarial_prediction)

            if count == final_batch:
                acc = correct / total * 100
                accuracy.append(acc)
                correct = 0
                total = 0
                first_num.append(len(adversarial_examples))
                break
            else:
                count += 1

    plot_adversarial_accuracy('Fast Gradient Sign Method', epsilons, accuracy)

    # Convert to tensor
    adversarial_examples, original_labels, adversarial_labels = _to_tensor(adversarial_examples, original_labels,
                                                                           adversarial_labels)

    return adversarial_examples, original_labels, adversarial_labels


def _to_tensor(adversarial_examples, original_labels, adversarial_labels):
    adversarial_examples = torch.cat([torch.from_numpy(ex) for ex in adversarial_examples], dim=0)
    original_labels = torch.cat([torch.from_numpy(lbl) for lbl in original_labels], dim=0)
    adversarial_labels = torch.cat([torch.from_numpy(lbl) for lbl in adversarial_labels], dim=0)

    original_labels = torch.argmax(original_labels, dim=1)
    adversarial_labels = torch.argmax(adversarial_labels, dim=1)
    return adversarial_examples, original_labels, adversarial_labels
