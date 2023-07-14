import torch


def train_model(model, num_epochs, train_loader, optimizer, device, criterion):
    correct = 0
    total = 0

    for epoch in range(num_epochs):
        running_loss = 0.0

        # index represents the looping variable, or the batch index
        # date represents the batch of training samples and their corresponding labels
        # The 0 specifies the starting value for the index
        for index, data in enumerate(train_loader, 0):

            # Separating the input images and labels into their own variables
            inputs, labels = data

            optimizer.zero_grad()
            outputs = model(inputs.to(device))
            loss = criterion(outputs, labels.to(device))
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            # Calculate the accuracy
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels.to(device)).sum().item()

            if index % 200 == 199:
                print('[%d, %5d] loss: %.3f, accuracy: %.2f %%' % (epoch + 1, index + 1,
                                                                   running_loss / 200, 100 * correct / total))
                running_loss = 0.0

    print(f'Total: {total}\tCorrect: {correct}')
    print('Training finished.')
