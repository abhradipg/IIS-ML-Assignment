from smallNet import *

from tqdm import tqdm
from torch import optim
from torchvision import datasets, transforms
from utils import test,load_data
import tracemalloc
import time
import os
from torch.profiler import profile, record_function, ProfilerActivity

def main():
    device = torch.device("cpu:0")
    model = SmallNet()

    model.qconfig = torch.quantization.get_default_qconfig('fbgemm')
    model = torch.quantization.fuse_modules(model, [['conv1', 'relu1'], ['conv2', 'relu2']])
    model = torch.quantization.prepare(model)

    model = torch.quantization.convert(model)

    # loading your quantization model
    model.load_state_dict(torch.load("mnist_cnn_int8.pt"))
    model=model.to(device)
    

    batch_size = 64
    use_cuda = torch.cuda.is_available()
    train_loader, test_loader = load_data(batch_size, use_cuda)
    data, label = next(iter(train_loader))

    #measuring peak memory on inference on a batch of 64
    tracemalloc.start()
    output = model(data)
    size, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    print(f"Peak allocated memory={peak}")

    #measuring execution time
    #start=time.time()
    #test(model, device, test_loader)
    #end=time.time()
    #print(f"Execution Time = {end - start}")

    #measuring execution time
    with profile(activities=[ProfilerActivity.CPU],record_shapes=True) as prof:
        test(model, device, test_loader)


    print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=100))


    #measuring compression ratio
    original = os.path.getsize("mnist_cnn.pt")/1e3
    quantized = os.path.getsize("mnist_cnn_int8.pt")/1e3
    print(f"Size of original file = {original}KB")
    print(f"Size of quantized file = {quantized}KB")
    print(f"Compression ratio = {original/quantized}")
    return 0


if __name__ == "__main__":
    main()