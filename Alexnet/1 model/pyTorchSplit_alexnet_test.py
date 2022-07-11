# FOR MYSTIKOS SGX IMPLEMENTATION
from re import sub
from torchvision import models
import torch
import os
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision import transforms
from PIL import Image
import os, random, struct, binascii
import time
import gc
import functools
from sys import getrefcount

# prepare class Flatten to avoid pyTorch model bug
class Flatten(nn.Module):
    def __init__(self):
        super(Flatten, self).__init__()
        
    def forward(self, x):
        x = x.view(x.size(0), -1)
        return x

# Start of execution time
start_time = time.perf_counter()

# Prepare a transform to get the input image into a format (e.g., x,y dimensions) the classifier
# expects.

transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(
    mean=[0.485, 0.456, 0.406],
    std=[0.229, 0.224, 0.225]
)])

# Load the image.
img = Image.open("input.jpg")

# Apply the transform to the image.
img_t = transform(img)
# Returns a new tensor with a dimension of size specified
batch_t = torch.unsqueeze(img_t, 0)

# intermediate result to be stored in a single variable

# lru cache decorater: wraps a function with a memoizing callable that saves up to maxsize of most recent calls.
# if Maxsize is set to None, LRU is disabled. Cache can grow without bond
# @functools.lru_cache(maxsize=None)
def tempModel(out, model):
    # set model to inference mode
    model.eval()
    out = model(out)
    # deletes intermediary memory address used to store model
    # and removes reference to submodel
    # even without deleting 'model' during execution, reference count for submodel will increase, and after execution will decrease back to its original value
    del model
    # only keep 'out' variable
    return out

# instead of reuploading the model, only load model into variable when required
submodel = torch.load("alexnet-pretrained.pt")
out = tempModel(batch_t, submodel)
print("Start of program")
# print("submodel ref count: ", getrefcount(submodel))
# print("out ref count: ", getrefcount(out))
print("-----------")
# clears and invalidates cache inside tempModel function
# tempModel.cache_clear()
# del submodel
# once submodel has been deleted, virtual memory adress will have no assignment to the variable
# submodel is now undefined
# print(hex(id(submodel))) <-- will not be able print address as variable is not defined



# Load the classes from disk.
with open('classes.txt') as f:
    classes = [line.strip() for line in f.readlines()]

# # Sort the predictions.
_, indices = torch.sort(out, descending=True)

# Convert into percentages.
percentage = torch.nn.functional.softmax(out, dim=1)[0] * 100


print("End of program")

submodel = None 
out = None

# collect all trash objects (objects values that have been dereferenced)
gc.collect()

inference_time = time.perf_counter()
print("End of inference time: ", inference_time - start_time, "seconds")

print("\n----------Inferencing Completed----------\n")


# Print the 5 most likely predictions.
with open("result.txt", "w") as outfile:
    outfile.write(str([(classes[idx], percentage[idx].item()) for idx in indices[0][:5]]))

end_time = time.perf_counter()
print("End of execution time: ", end_time - start_time, "seconds")



# 2nd inference
transform_2 = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(
    mean=[0.485, 0.456, 0.406],
    std=[0.229, 0.224, 0.225]
)])

# Load the image.
img_2 = Image.open("input.jpg")

# Apply the transform to the image.
img_t_2 = transform_2(img_2)
# Returns a new tensor with a dimension of size specified
batch_t_2 = torch.unsqueeze(img_t_2, 0)

def tempModel_2(out, model):
    # set model to inference mode
    model.eval()
    out = model(out)
    # deletes intermediary memory address used to store model
    # and removes reference to submodel
    # even without deleting 'model' during execution, reference count for submodel will increase, and after execution will decrease back to its original value
    del model
    # only keep 'out' variable
    return out

submodel_2 = torch.load("alexnet-pretrained.pt")
out_2 = tempModel_2(batch_t_2, submodel_2)
print("Start of program")

print("-----------")

second_inference_time = time.perf_counter()
print("End of second inference time from start time: ", second_inference_time - start_time, "seconds")

final_end_time = second_inference_time - end_time
print("Isolated runtime of single inference execution: ", final_end_time, "seconds")