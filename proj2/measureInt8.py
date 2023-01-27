from smallNet import *

from tqdm import tqdm
from torch import optim
from torchvision import datasets, transforms
from utils import test


def main():
    model = SmallNet()

    model.qconfig = torch.quantization.get_default_qconfig('fbgemm')
    model_fp32_fused = torch.quantization.fuse_modules(model, [['conv1', 'relu1'], ['conv2', 'relu2']])
    model_fp32_prepared = torch.quantization.prepare(model_fp32_fused)

    model_int8 = torch.quantization.convert(model_fp32_prepared)

    # loading your quantization model
    model_int8.load_state_dict(torch.load("mnist_cnn_int8.pt"))


    # do your measurements


    return 0


if __name__ == "__main__":
    main()