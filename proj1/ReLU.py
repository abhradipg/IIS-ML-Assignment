import torch
import torch.nn as nn
import time


def main(args):
    inpSize = (16, 64, 112, 112)
    device = torch.device("cuda:6")
    num_of_conv = 10
    num_of_fc = 1
    starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)

    l = [
       nn.Conv2d(64 , 64, kernel_size=3, padding=1) for i in range(num_of_conv)
    ]

    l += [nn.MaxPool2d(kernel_size=4, stride=4), nn.Flatten(), nn.Linear(50176, 4096)]

    l += [
        nn.Linear(4096, 4096) for i in range(num_of_fc)
    ]


    model = nn.Sequential(*l).to(device)
    x = torch.randn(inpSize).to(device)
    y = model(x)
    torch.cuda.synchronize()
    print("sleeping")
    time.sleep(5)
    print("start")
    starter.record()
    y = model(x)
    ender.record()
    torch.cuda.synchronize()
    curr_time = starter.elapsed_time(ender)
    print(curr_time)
    print(y.size())

    return 0


if __name__ == "__main__":
    main(None)
