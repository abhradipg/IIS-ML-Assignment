import torch
import torch.nn as nn


def main(args):
    inpSize = (16, 64, 112, 112)
    device1 = torch.device("cuda:7")
    device2 = torch.device("cuda:5")
    num_of_conv1 = 350
    num_of_conv2 = 150
    num_of_fc1 = 250
    num_of_fc2 = 0
    starter1, ender1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    starter2, ender2 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)

    l1 = [
        nn.Conv2d(64, 64, kernel_size=3, padding=1) for i in range(num_of_conv1)
    ]

    l2 = [
       nn.Conv2d(64 , 64, kernel_size=3, padding=1) for i in range(num_of_conv2)
    ]
 
    l2 += [nn.MaxPool2d(kernel_size=4, stride=4), nn.Flatten(), nn.Linear(50176, 4096)]

    l2 += [
        nn.Linear(4096, 4096) for i in range(num_of_fc1)
    ]




    model1 = nn.Sequential(*l1).to(device1)

    model2 = nn.Sequential(*l2).to(device2)

    x = torch.randn(inpSize).to(device1)
    y = model1(x)
    torch.cuda.synchronize()
    starter1.record()
    y = model1(x)
    ender1.record()
    torch.cuda.synchronize()
    curr_time = starter1.elapsed_time(ender1)
    print(curr_time)
    y2 = y.to(device2)
    z = model2(y2)
    torch.cuda.synchronize()
    starter2.record()
    z = model2(y2)
    ender2.record()
    torch.cuda.synchronize()
    curr_time = starter2.elapsed_time(ender2)
    print(curr_time)
    print(z.size())

    return 0


if __name__ == "__main__":
    main(None)
