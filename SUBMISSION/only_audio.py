# -*- coding: utf-8 -*-
"""
Created on Sun Apr 21 23:14:24 2019

@author: ragga
This file is used for training and testing only audio dataset
"""

from __future__ import print_function, division
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision as tv
import torch.optim as optim
import numpy as np


    
#aud_iter = iter(aud_data_loader['audio_train'])

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.aud_conv1 = nn.Conv1d(1, 16, 3)
        self.aud_pool = nn.MaxPool1d(2)
        self.aud_conv2 = nn.Conv1d(16, 32, 3)
        self.aud_conv3 = nn.Conv1d(32, 64, 3)
        self.fc1 = nn.Linear(64*15, 120)
        self.d1 = nn.Dropout(0.2)
        self.bn1 = nn.BatchNorm1d(64*15)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 8)
        
    def forward(self, x_aud):
        x_aud = self.aud_pool(F.relu(self.aud_conv1(x_aud)))
        x_aud = F.relu(self.aud_conv2(x_aud))
        x_aud = F.relu(self.aud_conv3(x_aud))
        x_aud = x_aud.view(-1, 64*15)
        x_aud = self.bn1(x_aud)
        x_aud = self.d1(x_aud)
        x = F.relu(self.fc1(x_aud))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

def train(aud_data_loader, criterion, net, device, optimizer):
    t_acc = []
    t_loss = []
    v_acc = []
    v_loss = []
    for epoch in range (30):
        running_loss = 0.0
        l = 0
        total = 0
        correct = 0
        for i, data in enumerate(aud_data_loader['aud_train']):
            # get the inputs
            aud_inputs, aud_labels = data
            aud_inputs = aud_inputs.type(torch.FloatTensor)
            aud_inputs, aud_labels = aud_inputs.to(device), aud_labels.to(device)
            # zero the parameter gradients
            optimizer.zero_grad()
            outputs = net(aud_inputs)
            loss = criterion(outputs, aud_labels)
            _, predicted = torch.max(outputs.data, 1)
            total += aud_labels.size(0)
            correct += (predicted == aud_labels).sum().item()
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            l += loss.item()
            if i % 2 == 1:    # print every 2000 mini-batches
                print('[%d, %5d] loss: %.3f' %
                      (epoch + 1, i + 1, running_loss / 2))
                running_loss = 0.0
        
        t_loss.append(l)
        t_acc.append(100*correct/total)
        total = 0
        correct = 0
        l = 0
        with torch.no_grad():
            for i, data in enumerate(aud_data_loader['aud_val'], 0):
                aud_inputs, aud_labels = data
                aud_inputs = aud_inputs.type(torch.FloatTensor)
                aud_inputs, aud_labels = aud_inputs.to(device), aud_labels.to(device)
                outputs = net(aud_inputs)
                loss = criterion(outputs, aud_labels)
                l += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += aud_labels.size(0)
                correct += (predicted == aud_labels).sum().item()
    
        v_loss.append(l)
        v_acc.append(100*correct/total)
        print("Accuracy of the network on validation set is : %d %%" %(100 *correct/total))
    
    print('Finished Training')
    np.save('ONLY_AUDIO_VAL_LOSS', v_loss)
    np.save('ONLY_AUDIO_VAL_ACC', v_acc)
    np.save('ONLY_AUDIO_TRAIN_LOSS', t_loss)
    np.save('ONLY_AUDIO_TRAIN_ACC', t_acc)


def test(aud_data_loader, criterion, net, device, optimizer):
    correct = 0
    total = 0
    nb_classes = 8
    confusion_matrix = torch.zeros(nb_classes, nb_classes)
    with torch.no_grad():
        for i, data in enumerate(aud_data_loader['aud_test'], 0):
                aud_inputs, aud_labels = data
                aud_inputs = aud_inputs.type(torch.FloatTensor)
                aud_inputs, aud_labels = aud_inputs.to(device), aud_labels.to(device)
                outputs = net(aud_inputs)
                _, predicted = torch.max(outputs.data, 1)
                total += aud_labels.size(0)
                correct += (predicted == aud_labels).sum().item()
                for t, p in zip(aud_labels.view(-1), predicted.view(-1)):
                    confusion_matrix[t.long(), p.long()] += 1

    np.save('ONLY_AUDIO_CONFUSION_MATRIX', confusion_matrix)
    print('Accuracy of the network on the test images: %d %%' % (
            100 * correct / total))

def npy_loader(path):
    sample = torch.from_numpy(np.load(path))
    return sample


folder = 'C:/Carnegie Mellon/10707-DL/Project/tmpdir'
def only_audio(folder):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print (device)

    aud_dataset = ['aud_train', 'aud_test', 'aud_val']

    audio_data = {}

    print(folder)
    for x in aud_dataset:
        audio_data[x] = tv.datasets.DatasetFolder(root=folder + '/' + x, loader=npy_loader, extensions=['.npy'])
      
    aud_data_loader = {}
    for x in aud_dataset:
        aud_data_loader[x] = torch.utils.data.DataLoader(audio_data[x], batch_size=32, 
                   shuffle=True, num_workers=0)
    net = Net()

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
    train(aud_data_loader, criterion, net, device, optimizer)
    test(aud_data_loader, criterion, net, device, optimizer)

if __name__ == "__main__":
    only_audio(folder)