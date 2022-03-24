from torchvision import models
import torch

alexnet = models.alexnet(pretrained=True)
torch.save(alexnet, "alexnet-pretrained.pt")

resnet50 = models.resnet50(pretrained=True)
torch.save(resnet50, "resnet50.pt")

resnet101 = models.resnet101(pretrained=True)
torch.save(resnet101, "resnet101.pt")

vgg16 = models.vgg16(pretrained=True)
torch.save(vgg16, "vgg16.pt")

vgg19 = models.vgg19(pretrained=True)
torch.save(vgg19, "vgg19.pt")

mobilenet_v3_small = models.mobilenet_v3_small(pretrained=True)
torch.save(mobilenet_v3_small, "mobilenetV3_small.pt")

mobilenet_v3_large = models.mobilenet_v3_large(pretrained=True)
torch.save(mobilenet_v3_large, "mobilenetV3_large.pt")

squeezenet1_0 = models.squeezenet1_0(pretrained=True)
torch.save(squeezenet1_0, "squeezenet.pt")

print("Pre-trained models have been saved!")