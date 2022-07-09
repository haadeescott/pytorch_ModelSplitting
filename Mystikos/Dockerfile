FROM python:3.9-slim

RUN pip3 install torchvision

WORKDIR /app
COPY pyTorchSplit_alexnet.py /app
COPY alexnet-pretrained.pt /app
COPY input.jpg /app
COPY classes.txt /app
COPY result_alexnet.txt /app

ENTRYPOINT ["python3", "pyTorchSplit_alexnet.py"]
