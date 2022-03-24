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
submodel_1 = torch.load("submodel_1.pt")
submodel_2 = torch.load("submodel_2.pt")
submodel_3 = torch.load("submodel_3.pt")
submodel_4 = torch.load("submodel_4.pt")
submodel_5 = torch.load("submodel_5.pt")
submodel_6 = torch.load("submodel_6.pt")
submodel_7 = torch.load("submodel_7.pt")
submodel_8 = torch.load("submodel_8.pt")
submodel_9 = torch.load("submodel_9.pt")
submodel_10 = torch.load("submodel_10.pt")
submodel_11 = torch.load("submodel_11.pt")
Main_Submodel2 = torch.load("Main_Submodel2.pt")

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
submodel_1.eval()
submodel_2.eval()
submodel_3.eval()
submodel_4.eval()
submodel_5.eval()
submodel_6.eval()
submodel_7.eval()
submodel_8.eval()
submodel_9.eval()
submodel_10.eval()
submodel_11.eval()
Main_Submodel2.eval()

output_submodel_1 = submodel_1(batch_t)
output_submodel_2 = submodel_2(output_submodel_1)
output_submodel_3 = submodel_3(output_submodel_2)
output_submodel_4 = submodel_4(output_submodel_3)
output_submodel_5 = submodel_5(output_submodel_4)
output_submodel_6 = submodel_6(output_submodel_5)
output_submodel_7 = submodel_7(output_submodel_6)
output_submodel_8 = submodel_8(output_submodel_7)
output_submodel_9 = submodel_9(output_submodel_8)
output_submodel_10 = submodel_10(output_submodel_9)
output_submodel_11 = submodel_11(output_submodel_10)
out = Main_Submodel2(output_submodel_11)

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
with open("result_alexnet.txt", "w") as outfile:
    outfile.write(str([(classes[idx], percentage[idx].item()) for idx in indices[0][:5]]))

end_time = time.perf_counter()
print("End of execution time: ", end_time - start_time, "seconds")
