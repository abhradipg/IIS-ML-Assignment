from smallNet import *

from tqdm import tqdm
from torch import optim
from torchvision import datasets, transforms
from utils import test

def main():
    device = torch.device("cpu:0")
    batch_size = 64

    # load float32 model 
    model = SmallNet().to(device)
    model.load_state_dict(torch.load("mnist_cnn.pt"))

    # do your measurements




if __name__ == "__main__":
    main()