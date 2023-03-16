from smallNet import *

from torch import optim
from utils import test, load_data


def main():
    batch_size = 64
    epochs = 10
    lr = 0.01
    momentum = 0.5
    seed = 1
    
    torch.manual_seed(seed)
    use_cuda = torch.cuda.is_available()


    device = torch.device("cuda" if use_cuda else "cpu")

    # call load_data
    train_loader, test_loader = load_data(batch_size, use_cuda)
    

    model = SmallNet().to(device)
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum)
    criterion = nn.CrossEntropyLoss()
    
    # your training loop for 10 epoch
    for epoch in range(epochs):  # loop over the dataset multiple times

         for inputs, labels in train_loader:
             # get the inputs; data is a list of [inputs, labels]
             inputs = inputs.to(device)
             labels = labels.to(device)

             # zero the parameter gradients
             optimizer.zero_grad()

             # forward + backward + optimize
             outputs = model(inputs)
             loss = criterion(outputs, labels)
             loss.backward()
             optimizer.step()

    # save the model
    torch.save(model.state_dict(),"mnist_cnn.pt")
    test(model,device,test_loader)
    return model


if __name__ == "__main__":
    main()