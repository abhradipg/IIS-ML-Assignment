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

  

    # fusion command is provided for your beneift
    # model_fp32_fused = torch.quantization.fuse_modules(model, [['conv1', 'relu1'], ['conv2', 'relu2']])

    

    
    
    # Be sure to save your final model
    # torch.save(model_int8.state_dict(), "mnist_cnn_int8.pt")

    # Be sure to test your final model
    # 
    # test(model_int8, device, train_loader)
    return 0


if __name__ == "__main__":
    main()