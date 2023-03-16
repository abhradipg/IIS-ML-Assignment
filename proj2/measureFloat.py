from smallNet import *

from tqdm import tqdm
from torch import optim
from torchvision import datasets, transforms
from utils import test,load_data
import tracemalloc
import time
from torch.profiler import profile, record_function, ProfilerActivity

def main():
    device = torch.device("cpu:0")
    batch_size = 64

    # load float32 model 
    model = SmallNet().to(device)
    model.load_state_dict(torch.load("mnist_cnn.pt"))

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


if __name__ == "__main__":
    main()