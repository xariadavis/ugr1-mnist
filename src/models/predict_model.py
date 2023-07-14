import torch


# Output results after evaluating model on set
def evaluate_model(model, loader):
    model.eval()

    correct = 0
    total = 0
    predictions = []

    with torch.no_grad():
        for data in loader:
            inputs, labels = data

            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)

            correct += (predicted == labels).sum().item()
            predictions.append(predicted)

    predictions_array = torch.cat(predictions, 0)
    accuracy = (correct / total) * 100

    # Return accuracy, predictions_array
    return accuracy, predictions_array
