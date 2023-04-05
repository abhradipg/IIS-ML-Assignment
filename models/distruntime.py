import torch
from torch.utils.data.sampler import RandomSampler, SequentialSampler
import torchvision
import random
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import numpy as np
import torchmetrics
import torch.optim as optim
import argparse
import os
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler

class VGG(nn.Module):
    def __init__(self):
        super(VGG, self).__init__()
        # a vector descriping VGG16 dimensions
        vec = [
            64,
            64,
            "M",
            128,
            128,
            "M",
            256,
            256,
            256,
            256,
            "M",
            512,
            512,
            512,
            512,
            "M",
            512,
            512,
            512,
            512,
            "M",
        ]

        self.listModule = []
        in_channels = 3

        for v in vec:
            if v == "M":
                self.listModule += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                v = int(v)
                conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
                self.listModule += [conv2d, nn.ReLU(inplace=True)]
                in_channels = v

        self.features = nn.Sequential(*self.listModule)

        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
        
        self.cl = [
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 1000),
        ]
        self.classifier = nn.Sequential(*self.cl)

    
    def forward(self, x):
        # x = self.features(x)
        x = self.features(x)
        x = self.avgpool(x)
        x = x.flatten(1)
        x = self.classifier(x)
        return x


def init_distributed(args):
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
      args.world_size = int(os.environ['WORLD_SIZE'])
      args.gpu = int(os.environ['LOCAL_RANK'])+1
    else:
        print('Not using distributed mode')
        args.gpu = 0
        args.distributed = False
        return

    args.distributed = True
    torch.cuda.set_device(args.gpu)

    os.environ['MASTER_ADDR']= '127.0.0.1'
    os.environ['MASTER_PORT']= '29601'
    dist.init_process_group(backend='nccl', world_size=args.world_size)
    dist.barrier()


class MyDataset():
  def __init__(self):
    super(MyDataset, self).__init__()
    dataSize = (1024, 3,224 ,224 )
    self.inputs=torch.randn(dataSize)
    self.labels = []
    for i in range(1024):
       self.labels+=[random.randint(0,999)]

  def __len__(self):
    return len(self.inputs)  # number of samples in the dataset

  def __getitem__(self, index):
    return self.inputs[index], self.labels[index]


def main(args):
    batch_size = 4
    Dataset=MyDataset()
    init_distributed(args)
    sampler_train = DistributedSampler(Dataset, shuffle=True)
    trainloader = torch.utils.data.DataLoader(Dataset, batch_size, sampler=sampler_train)

    model = VGG().to(args.gpu)
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

    starter1, ender1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)

    torch.cuda.synchronize()
    starter1.record()
    for epoch in range(4):  # loop over the dataset multiple times

         sampler_train.set_epoch(epoch)
         running_loss = 0.0
         for i, data in enumerate(trainloader, 0):
             # get the inputs; data is a list of [inputs, labels]
             inputs, labels = data
             inputs = inputs.to(args.gpu)
             labels = labels.to(args.gpu)

             # zero the parameter gradients
             optimizer.zero_grad()

              # forward + backward + optimize
             outputs = model(inputs)
             loss = criterion(outputs, labels)
             loss.backward()
             optimizer.step()


    ender1.record()
    torch.cuda.synchronize()
    print(starter1.elapsed_time(ender1))
    print('Finished Training')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Required positional argument
    parser.add_argument("--world_size", help="Total number of nodes")

    args = parser.parse_args()
    main(args)
