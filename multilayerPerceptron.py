import argparse
import time
import os
import math
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import matplotlib.pyplot as plt


def parseTxt(txt_file):
    coordinates = []
    labels = []
    with open(txt_file, 'r') as f:
        for line in f:
            l = line.split()
            coordinates.append([float(l[0]), float(l[1])])
            labels.append([float(l[2])])
    
    # n_samples * 2, n_samples * 1
    return coordinates, labels


class Model(nn.Module):
    def __init__(self, n_hid):
        super().__init__()
        self.hid1 = nn.Linear(2, n_hid)
        self.activ1 = nn.Tanh() # [-1, 1]
        self.out = nn.Linear(n_hid, 1)
        self.activ2 = nn.Sigmoid() # [0, 1], >=0.5 means classify as 1
    
    def forward(self, input):
        polar_input = []
        batch_size = input.size()[0]
        for i in range(batch_size):
            x, y = input[i][0].item(), input[i][1].item()
            r = math.sqrt(x**2 + y**2)
            theta = math.atan2(y, x)
            polar_input.append([r, theta])
        
        tensor_input = torch.tensor(polar_input, dtype=torch.float)
        a = self.activ1(self.hid1(tensor_input))
        a = self.activ2(self.out(a))
        return a
            

def train(n_epochs, model, loss_fn, optimizer, train_loader, val_loader, device):
    model.to(device=device)
    for epoch in range(1, n_epochs + 1):
        loss_train_list = []
        for x_train, y_train in train_loader:
            x_train = x_train.to(device=device)
            y_train = y_train.to(device=device)
            y_pred_train = model(x_train) # batch_size * 1

            loss_train = loss_fn(y_pred_train, y_train) # BCELoss
            loss_train_list.append(loss_train.item())

            optimizer.zero_grad()
            loss_train.backward()
            optimizer.step()
        if epoch == 1 or epoch % 100 == 0:
            with torch.no_grad():
                loss_train_avg = np.array(loss_train_list).mean()
                loss_val_list = []
                total, correct = 0, 0
                for x_val, y_val in val_loader:
                    x_val = x_val.to(device=device)
                    y_val = y_val.to(device=device)
                    y_pred_val = model(x_val)

                    loss_val = loss_fn(y_pred_val, y_val)
                    loss_val_list.append(loss_val.item())

                    total += y_val.size()[0]
                    correct += ((y_pred_val >= 0.5) == y_val).sum().item()
                
                loss_val_avg = np.array(loss_val_list).mean()
                pred_accuracy = correct / total
            print('Epoch {:5d}: train loss {:.4f}, validation loss {:.4f}, prediction accuracy {:.4f}'.format(epoch, loss_train_avg, loss_val_avg, pred_accuracy))
        if(pred_accuracy == 1):
            break


def plotPrediction(model, txt_file):
    fig, ax = plt.subplots()
    xrange = torch.arange(start=-7, end=7.01, step=0.01, dtype=torch.float)
    yrange = torch.arange(start=-7, end=7.01, step=0.01, dtype=torch.float)
    xcoor = xrange.repeat(yrange.size()[0])
    ycoor = torch.repeat_interleave(yrange, xrange.size()[0])
    coors = torch.cat((xcoor.unsqueeze(1), ycoor.unsqueeze(1)), dim=1)
    with torch.no_grad():
        # model.eval() # toggle batch norm, dropout
        pred = (model(coors) >= 0.5).int()
    ax.pcolormesh(xrange, yrange, pred.view(xrange.size()[0], yrange.size()[0]), cmap='Pastel2')

    xw, yw, xb, yb = [], [], [], []
    with open(txt_file, 'r') as f:
        for line in f:
            l = line.split()
            if(l[2] == '0.0'):
                xb.append(float(l[0]))
                yb.append(float(l[1]))
            else:
                xw.append(float(l[0]))
                yw.append(float(l[1]))

    ax.scatter(xb, yb, c='k')
    ax.scatter(xw, yw, c='r')

    fig.savefig(os.path.join(save_path, 'prediction_graph.png'))

# 193 lines
txt_file = r'./two-spiral.txt'
save_path = r'.'

# device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
device = torch.device('cpu')

parser = argparse.ArgumentParser()
parser.add_argument('--epoch', type=str, default=10000, help='Max training epochs')
parser.add_argument('--hid', type=int, default=16, help='Number of hidden layers')
parser.add_argument('--lr', type=float, default=1e-2, help='Learning rate')
args = parser.parse_args()

coordinates, labels = parseTxt(txt_file)
coordinates, labels = torch.tensor(coordinates, dtype=torch.float), torch.tensor(labels, dtype=torch.float)

n_samples = labels.shape[0] # 193
n_val = round(0.2 * n_samples) # 39
shuffled_indices = torch.randperm(n_samples)
train_indices = shuffled_indices[:-n_val]
val_indices = shuffled_indices[-n_val:]
train_data = coordinates[train_indices]
train_label = labels[train_indices]
val_data = coordinates[val_indices]
val_label = labels[val_indices]
print('Dataset size:', n_samples)
print('Training set size:', train_label.size()[0])
print('Validation set size:', val_label.size()[0])

train_dataset = data.TensorDataset(train_data, train_label)
val_dataset = data.TensorDataset(val_data, val_label)
train_loader = data.DataLoader(train_dataset, batch_size=97, shuffle=True)
val_loader = data.DataLoader(val_dataset, batch_size=20, shuffle=True)

model = Model(args.hid)
loss_fn = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), eps=0.000001, lr=args.lr, betas=(0.9, 0.999), weight_decay=0.0001)

start_time = time.time()
train(args.epoch, model, loss_fn, optimizer, train_loader, val_loader, device)
print('Training time: {:.2f} seconds'.format(time.time() - start_time))
torch.save(model.state_dict(), os.path.join(save_path, 'twoSpiral_trained_model.pt'))
plotPrediction(model, txt_file)
