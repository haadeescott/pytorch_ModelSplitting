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
from Crypto.Cipher import AES
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

def getPass(path):
    with open(path) as f:
        pwd = f.readlines()
        aesSecretDecryption = pwd[0]
        f.close()
        return aesSecretDecryption

def decrypt_file(key, in_filename, out_filename, chunksize=24*1024):
    #  Decrypts a file using AES (CBC mode) with the given key

    with open(in_filename, 'rb') as infile:
        origsize = struct.unpack('<Q', infile.read(struct.calcsize('Q')))[0]
        iv = infile.read(16)
        decryptor = AES.new(key.encode("utf8"), AES.MODE_CBC, iv)

        with open(out_filename, 'wb') as outfile:
            while True:
                chunk = infile.read(chunksize)
                if len(chunk) == 0:
                    break
                outfile.write(decryptor.decrypt(chunk))

            outfile.truncate(origsize)

print(" ")

# specify which key to use to decrypt encrypted submodel
# this assumes all submodels are encrypted sequentially
print("Decrypting model...")
decrypt_file(getPass('secrets/secret_1.txt'), 'submodel_1_enc.pt', 'submodel_1.pt')
decrypt_file(getPass('secrets/secret_2.txt'), 'submodel_2_enc.pt', 'submodel_2.pt')
decrypt_file(getPass('secrets/secret_3.txt'), 'submodel_3_enc.pt', 'submodel_3.pt')
decrypt_file(getPass('secrets/secret_4.txt'), 'submodel_4_enc.pt', 'submodel_4.pt')
decrypt_file(getPass('secrets/secret_5.txt'), 'submodel_5_enc.pt', 'submodel_5.pt')
decrypt_file(getPass('secrets/secret_6.txt'), 'submodel_6_enc.pt', 'submodel_6.pt')
decrypt_file(getPass('secrets/secret_7.txt'), 'submodel_7_enc.pt', 'submodel_7.pt')
decrypt_file(getPass('secrets/secret_8.txt'), 'submodel_8_enc.pt', 'submodel_8.pt')
decrypt_file(getPass('secrets/secret_9.txt'), 'submodel_9_enc.pt', 'submodel_9.pt')
decrypt_file(getPass('secrets/secret_10.txt'), 'submodel_10_enc.pt', 'submodel_10.pt')
decrypt_file(getPass('secrets/secret_11.txt'), 'submodel_11_enc.pt', 'submodel_11.pt')
decrypt_file(getPass('secrets/secret_12.txt'), 'Main_Submodel2_enc.pt', 'Main_Submodel2.pt')
print("Model decrypted!")
print(" ")

# instead of reuploading the model, only load model into variable when required
# intermediate result to be stored in a single variable
submodel = torch.load("submodel_1.pt")
out = tempModel(batch_t, submodel)

print("Start of program")
print("-----------")

submodel = torch.load("submodel_2.pt")
out = tempModel(out, submodel)

submodel = torch.load("submodel_3.pt")
out = tempModel(out, submodel)

submodel = torch.load("submodel_4.pt")
out = tempModel(out, submodel)

submodel = torch.load("submodel_5.pt")
out = tempModel(out, submodel)

submodel = torch.load("submodel_6.pt")
out = tempModel(out, submodel)

submodel = torch.load("submodel_7.pt")
out = tempModel(out, submodel)

submodel = torch.load("submodel_8.pt")
out = tempModel(out, submodel)

submodel = torch.load("submodel_9.pt")
out = tempModel(out, submodel)

submodel = torch.load("submodel_10.pt")
out = tempModel(out, submodel)

submodel = torch.load("submodel_11.pt")
out = tempModel(out, submodel)

submodel = torch.load("Main_Submodel2.pt")
out = tempModel(out, submodel)

# Load the classes from disk.
with open('classes.txt') as f:
    classes = [line.strip() for line in f.readlines()]

# Sort the predictions.
_, indices = torch.sort(out, descending=True)

# Convert into percentages.
percentage = torch.nn.functional.softmax(out, dim=1)[0] * 100

print("End of program")

submodel = None 
out = None
# once submodel has been deleted, virtual memory adress will have no assignment to the variable
# submodel is now undefined

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

