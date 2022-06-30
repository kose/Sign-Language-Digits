#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse, os
import numpy as np
from sklearn.metrics import confusion_matrix

import torch
from torchvision import datasets, transforms
from dataset import MyDataset, Flip
from network import Net

##
## test main function
##
def test(args, model, device):

    dataset = MyDataset(csvfile="test.csv",  flip=Flip.both, transform=False, repeat=1)
    testloader = torch.utils.data.DataLoader(dataset, batch_size=10000, shuffle=False)
    
    data, labels_truth = testloader.__iter__().next()
    
    labels_truth = labels_truth.numpy().copy()

    _pred = model(data).numpy().copy()
    labels_pred = np.argmax(_pred, axis=1)

    result = confusion_matrix(labels_truth, labels_pred)

    print(result)

    ## accuracy

    accuracy = sum(labels_truth == labels_pred) / len(labels_truth)
    print("accuracy:", accuracy)
    
    # print("end")


##
## main function
##
def main():
# Testing settings
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--model', type=str, default="result/result.pt")
    parser.add_argument('--csv', type=str, default="result/result.csv")

    args = parser.parse_args()

    # modelname = os.path.join("result", args.model)
    modelname = args.model

    print(modelname)
    
    device = torch.device("cpu")

    model = Net()

    model.load_state_dict(torch.load(modelname))

    #
    with torch.no_grad():
        model.eval()
        test(args, model, device)


if __name__ == '__main__':
    main()

# import pdb; pdb.set_trace()

### Local Variables: ###
### truncate-lines:t ###
### End: ###
