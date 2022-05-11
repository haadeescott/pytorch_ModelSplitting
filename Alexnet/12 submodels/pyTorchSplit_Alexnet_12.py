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
 
# output_submodel_1  = submodel_1(batch_t)
# print("original:", output_submodel_1)
# # store intermediate result in memory
# # flush cache

# output_submodel_2 = submodel_2(output_submodel_1)
# temp_result = output_submodel_2

# # reload intermediate result
# # flush cache

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
submodel = torch.load("submodel_1.pt")
out = tempModel(batch_t, submodel)
print("Start of program")
print("submodel ref count: ", getrefcount(submodel))
print("out ref count: ", getrefcount(out))
print("-----------")
# clears and invalidates cache inside tempModel function
# tempModel.cache_clear()
# del submodel
# once submodel has been deleted, virtual memory adress will have no assignment to the variable
# submodel is now undefined
# print(hex(id(submodel))) <-- will not be able print address as variable is not defined

submodel = torch.load("submodel_2.pt")
# print(id(submodel))
out = tempModel(out, submodel)
# tempModel.cache_info()
# tempModel.cache_clear()
# del submodel

submodel = torch.load("submodel_3.pt")
out = tempModel(out, submodel)
# del submodel

submodel = torch.load("submodel_4.pt")
out = tempModel(out, submodel)
# del submodel

submodel = torch.load("submodel_5.pt")
out = tempModel(out, submodel)
# del submodel

submodel = torch.load("submodel_6.pt")
out = tempModel(out, submodel)
# del submodel

submodel = torch.load("submodel_7.pt")
out = tempModel(out, submodel)
# del submodel

submodel = torch.load("submodel_8.pt")
out = tempModel(out, submodel)
# del submodel

submodel = torch.load("submodel_9.pt")
out = tempModel(out, submodel)
# del submodel

submodel = torch.load("submodel_10.pt")
out = tempModel(out, submodel)
# del submodel

submodel = torch.load("submodel_11.pt")
out = tempModel(out, submodel)
# del submodel

submodel = torch.load("Main_Submodel2.pt")
out = tempModel(out, submodel)


# Load the classes from disk.
with open('classes.txt') as f:
    classes = [line.strip() for line in f.readlines()]

# Sort the predictions.
_, indices = torch.sort(out, descending=True)

# Convert into percentages.
percentage = torch.nn.functional.softmax(out, dim=1)[0] * 100
# del out
# gc.collect()

# Collect all objects inside the lru cache wrapper
# objects = [i for i in gc.get_objects() 
#            if isinstance(i, functools._lru_cache_wrapper)]
  
# Clear all objects inside objects
# for object in objects:
#     object.cache_clear()

print("End of program")
print("submodel ref count: ", getrefcount(submodel))
print("out ref count: ", getrefcount(out))
submodel = None 
out = None

# collect all trash objects (objects values that have been dereferenced)
gc.collect()

inference_time = time.perf_counter()
print("End of inference time: ", inference_time - start_time, "seconds")

print("\n----------Inferencing Completed----------\n")


# Print the 5 most likely predictions.
with open("result_alexnet.txt", "w") as outfile:
    outfile.write(str([(classes[idx], percentage[idx].item()) for idx in indices[0][:5]]))

end_time = time.perf_counter()
print("End of execution time: ", end_time - start_time, "seconds")
