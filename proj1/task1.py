import torch
import torch.nn as nn


def main(args):
    inpSize = (16, 64, 112, 112)
    device = torch.device("cuda:0")
    num_of_conv = 20
    num_of_fc = 20

    l = [
        nn.Conv2d(64, 64, kernel_size=3, padding=1) for i in range(num_of_conv)
    ]

    l += [nn.MaxPool2d(kernel_size=4, stride=4), nn.Flatten(), nn.Linear(50176, 4096)]

    l += [
        nn.Linear(4096, 4096) for i in range(num_of_fc-1)
    ]


    model = nn.Sequential(*l).to(device)

    x = torch.randn(inpSize).to(device)


    y = model(x)

    print(y.size())

    return 0


if __name__ == "__main__":
    main(None)