import subprocess
from threading import Thread , Timer
import time

def get_gpu_stat(command):
    output = subprocess.getoutput(command)
    print(output)


def print_gpu_stat(command):
    Timer(0.1, print_gpu_stat ,[command]).start()
    get_gpu_stat(command)


command = "nvidia-smi pmon -c 1"
print_gpu_stat(command)
