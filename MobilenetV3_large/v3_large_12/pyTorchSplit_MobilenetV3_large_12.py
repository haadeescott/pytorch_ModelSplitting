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

# prepare class Flatten to avoid pyTorch model bug
class Flatten(nn.Module):
    def __init__(self):
        super(Flatten, self).__init__()
        
    def forward(self, x):
        x = x.view(x.size(0), -1)
        return x

# Start of execution time
start_time = time.perf_counter()

# Load the model from a file
submodel_1_1_1 = torch.load("submodel_1_1_1.pt")
submodel_1_1_2 = torch.load("submodel_1_1_2.pt")
submodel_1_2_1 = torch.load("submodel_1_2_1.pt")
submodel_1_2_2 = torch.load("submodel_1_2_2.pt")
submodel_1_3_1 = torch.load("submodel_1_3_1.pt")
submodel_1_3_2 = torch.load("submodel_1_3_2.pt")
submodel_1_3_3 = torch.load("submodel_1_3_3.pt")
submodel_1_4 = torch.load("submodel_1_4.pt")
submodel_1_5 = torch.load("submodel_1_5.pt")
submodel_1_6 = torch.load("submodel_1_6.pt")
submodel_1_7 = torch.load("submodel_1_7.pt")
submodel_2 = torch.load("submodel_2.pt")

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

# Prepare the model and run the classifier.
submodel_1_1_1.eval()
submodel_1_1_2.eval()
submodel_1_2_1.eval()
submodel_1_2_2.eval()
submodel_1_3_1.eval()
submodel_1_3_2.eval()
submodel_1_3_3.eval()
submodel_1_4.eval()
submodel_1_5.eval()
submodel_1_6.eval()
submodel_1_7.eval()
submodel_2.eval()

output_submodel_1_1_1 = submodel_1_1_1(batch_t)
output_submodel_1_1_2 = submodel_1_1_2(output_submodel_1_1_1)
output_submodel_1_2_1 = submodel_1_2_1(output_submodel_1_1_2)
output_submodel_1_2_2 = submodel_1_2_2(output_submodel_1_2_1)
output_submodel_1_3_1 = submodel_1_3_1(output_submodel_1_2_2)
output_submodel_1_3_2 = submodel_1_3_2(output_submodel_1_3_1)
output_submodel_1_3_3 = submodel_1_3_3(output_submodel_1_3_2)
output_submodel_1_4 = submodel_1_4(output_submodel_1_3_3)
output_submodel_1_5 = submodel_1_5(output_submodel_1_4)
output_submodel_1_6 = submodel_1_6(output_submodel_1_5)
output_submodel_1_7 = submodel_1_7(output_submodel_1_6)
out = submodel_2(output_submodel_1_7)

# print('\nRAM memory % used:', psutil.virtual_memory()[2])
# Load the classes from disk.
with open('classes.txt') as f:
    classes = [line.strip() for line in f.readlines()]

# Sort the predictions.
_, indices = torch.sort(out, descending=True)

# Convert into percentages.
percentage = torch.nn.functional.softmax(out, dim=1)[0] * 100
inference_time = time.perf_counter()
print("End of inference time: ", inference_time - start_time, "seconds")

print("\n----------Inferencing Completed----------\n")


# Print the 5 most likely predictions.
with open("result_mobilenetV3_large.txt", "w") as outfile:
    outfile.write(str([(classes[idx], percentage[idx].item()) for idx in indices[0][:5]]))

end_time = time.perf_counter()
print("End of execution time: ", end_time - start_time, "seconds")
