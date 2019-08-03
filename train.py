import torch
import torch.nn as nn
import torchvision
from torchvision.transforms import transforms
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.autograd import Variable


img_width = 300
img_height = 300
class_count = 3 # gestures = 2, objects = 3
classifier_type = "objects" # "objects", "gestures"


class Net(nn.Module):
    def __init__(self, num_classes=class_count):
        super(Net, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=3, out_channels=24, kernel_size=3, stride=2, padding=1)
        self.relu1 = nn.ReLU()

        self.pool1 = nn.MaxPool2d(kernel_size=2)

        self.conv2 = nn.Conv2d(in_channels=24, out_channels=24, kernel_size=3, stride=1, padding=1)
        self.relu2 = nn.ReLU()

        self.conv3 = nn.Conv2d(in_channels=24, out_channels=24, kernel_size=3, stride=1, padding=1)
        self.relu3 = nn.ReLU()

        self.conv4 = nn.Conv2d(in_channels=24, out_channels=24, kernel_size=3, stride=1, padding=1)
        self.relu4 = nn.ReLU()

        self.pool2 = nn.MaxPool2d(kernel_size=2)

        self.dropout1 = nn.Dropout(0.3)

        self.conv5 = nn.Conv2d(in_channels=24, out_channels=24, kernel_size=3, stride=1, padding=1)
        self.relu5 = nn.ReLU()

        self.conv6 = nn.Conv2d(in_channels=24, out_channels=24, kernel_size=3, stride=1, padding=1)
        self.relu6 = nn.ReLU()

        self.conv7 = nn.Conv2d(in_channels=24, out_channels=24, kernel_size=3, stride=1, padding=1)
        self.relu7 = nn.ReLU()

        self.conv8 = nn.Conv2d(in_channels=24, out_channels=24, kernel_size=3, stride=1, padding=1)
        self.relu8 = nn.ReLU()

        self.pool3 = nn.MaxPool2d(kernel_size=2)

        self.fc1 = nn.Linear(in_features=24*18*18, out_features=128)
        self.relu9 = nn.ReLU()

        self.dropout2 = nn.Dropout(0.3)

        self.fc2= nn.Linear(in_features=128, out_features=num_classes)

    def forward(self, input):
        output = self.conv1(input)
        output = self.relu1(output)

        output = self.pool1(output)

        output = self.conv2(output)
        output = self.relu2(output)

        utput = self.conv3(output)
        output = self.relu3(output)

        utput = self.conv4(output)
        output = self.relu4(output)

        output = self.pool2(output)

        output = self.dropout1(output)

        output = self.conv5(output)
        output = self.relu5(output)

        output = self.conv6(output)
        output = self.relu6(output)

        output = self.conv7(output)
        output = self.relu7(output)

        output = self.conv8(output)
        output = self.relu8(output)

        output = self.pool3(output)

        output = output.view(-1, 24*18*18)

        output = self.fc1(output)
        output = self.relu9(output)

        output = self.dropout2(output)

        output = self.fc2(output)

        return output


def load_train_dataset(train_transformations):
    data_path = "data_{}/train/".format(classifier_type)

    train_dataset = torchvision.datasets.ImageFolder(
        root=data_path,
        transform=train_transformations
    )

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=32,
        num_workers=4,
        shuffle=True
    )

    return train_loader


def load_test_dataset(test_transformations):
    data_path = "data_{}/test/".format(classifier_type)

    test_dataset = torchvision.datasets.ImageFolder(
        root=data_path,
        transform=test_transformations
    )

    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=32,
        num_workers=4,
        shuffle=False
    )

    return test_loader


def test():
    model.eval()
    test_acc = 0.0
    for i, (images, labels) in enumerate(test_loader):
        if cuda_avail:
            images = Variable(images.cuda())
            labels = Variable(labels.cuda())

        # Predict classes using images from the test set.
        outputs = model(images)
        _, prediction = torch.max(outputs.data, 1)

        test_acc += torch.sum(prediction == labels.data)

    return test_acc


def train(num_epochs):
    best_acc = 0.0
    best_acc_train = 0.0

    for epoch in range(num_epochs):
        model.train()
        train_acc = 0.0
        train_loss = 0.0

        for i, (images, labels) in enumerate(train_loader):
            if cuda_avail:
                images = Variable(images.cuda())
                labels = Variable(labels.cuda())

            # Clear all accumulated gradients.
            optimizer.zero_grad()

            # Predict classes using images from the training set.
            outputs = model(images)

            # Compute the loss based on the predictions and actual labels
            loss = loss_fn(outputs, labels)

            # Backpropagate the loss
            loss.backward()

            # Adjust parameters according to the computed gradients
            optimizer.step()

            train_loss += loss.cpu().data * images.size(0)
            _, prediction = torch.max(outputs.data, 1)

            train_acc += torch.sum(prediction == labels.data)

        # Call the learning rate adjustment function
        adjust_learning_rate(epoch)

        # Evaluate on the test set
        test_acc = test()

        # Save the model if it is a new best for accuracy.
        if test_acc >= best_acc or train_acc >= best_acc_train:
            save_models(epoch, train_acc, test_acc)
            best_acc = test_acc
            best_acc_train = train_acc

        # Print metrics for epoch.
        print("Epoch {}, Train Accuracy: {} , TrainLoss: {} , Test Accuracy: {}".format(epoch, train_acc, train_loss,test_acc))


def adjust_learning_rate(epoch):
    lr = 0.000001

    if epoch > 180:
        lr = lr / 1000
    elif epoch > 150:
        lr = lr / 100
    elif epoch > 120:
        lr = lr / 10

    for param_group in optimizer.param_groups:
        param_group["lr"] = lr


def save_models(epoch, train_acc, test_acc):
    torch.save(model.state_dict(), "{}_{}_{}-{}.model".format(classifier_type, epoch, train_acc, test_acc))
    print("Model saved.")


if __name__ == "__main__":
    train_transformations = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    test_transformations = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    cuda_avail = torch.cuda.is_available()

    model = Net(num_classes=class_count)

    if cuda_avail:
        model.cuda()

    optimizer = Adam(model.parameters(), lr=0.000001, weight_decay=0.0001)
    loss_fn = nn.CrossEntropyLoss()

    train_loader = load_train_dataset(train_transformations)
    test_loader = load_test_dataset(test_transformations)

    print("Starting training.")
    train(500)
