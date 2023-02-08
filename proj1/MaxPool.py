import torch
import torch.nn as nn
import time
import subprocess
from threading import Thread , Timer
import time

def get_gpu_stat(command):
    output = subprocess.getoutput(command)
    print(output)


def print_gpu_stat(command):
    Timer(0.1, print_gpu_stat ,[command]).start()
    get_gpu_stat(command)


def main(args):
    inpSize = (64, 64, 224, 224)
    device = torch.device("cuda:0")
    starter = torch.cuda.Event(enable_timing=True)
    ender = torch.cuda.Event(enable_timing=True)

    command = "nvidia-smi dmon -o T -c 1| grep 0"
    print_gpu_stat(command)
    time.sleep(2);
    model = nn.MaxPool2d(kernel_size=3, padding=1).to(device)
    x = torch.randn(inpSize).to(device)
    y = model(x)
    torch.cuda.synchronize()
    time.sleep(2)
    starter.record()
    y = model(x)
    ender.record()
    torch.cuda.synchronize()
    curr_time = starter.elapsed_time(ender)
    time.sleep(2)
    print(curr_time)
    print(y.size())

    return 0


if __name__ == "__main__":
    main(None)


