from smallNet import *

from torch import optim
from utils import test, load_data, train


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
    
    
    # your training loop for 10 epoch
    
    # save the model
    torch.save(model.state_dict(),"mnist_cnn.pt")
    
    return model


if __name__ == "__main__":
    main()