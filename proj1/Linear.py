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
    Timer(0.05, print_gpu_stat ,[command]).start()
    get_gpu_stat(command)


def main(args):
    inpSize = (64, 512 * 7 * 7)
    device = torch.device("cuda:0")
    starter = torch.cuda.Event(enable_timing=True)
    ender = torch.cuda.Event(enable_timing=True)

    command = "nvidia-smi dmon -o T -c 1| grep 0"
    print_gpu_stat(command)
    time.sleep(2);
    model = nn.Linear(512 * 7 * 7, 4096).to(device)
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


