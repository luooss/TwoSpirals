import argparse
import math
import numpy as np
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from torch.nn.modules import activation


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


def divideData(coordinates, labels, subnum):
    subsets = []
    num_cases = labels.shape[0]
    sub_num_cases = math.ceil(num_cases / subnum)
    indices = np.random.permutation(num_cases)
    for i in range(0, num_cases, sub_num_cases):
        subsets.append([coordinates[indices[i:i+sub_num_cases]], labels[indices[i:i+sub_num_cases]]])
    
    print('Data is divided into {} subsets:'.format(len(subsets)), end=' ')
    for l in subsets:
        print(l[1].shape[0], end=' ')
    print()
    return subsets


def trainM3Net(submodule, subnum, lr, w_subsets, b_subsets, coordinates, labels):
    m3n = []
    xx, yy = np.meshgrid(np.arange(-7, 7.01, 0.01), np.arange(-7, 7.01, 0.01))
    cm = plt.cm.RdBu
    cm_points = ListedColormap(['#000000', '#FFFFFF'])
    fig, ax = plt.subplots(subnum, subnum, figsize=(9, 9))
    fig_min, ax_min = plt.subplots(subnum, 1, figsize=(3, 9))
    fig_max, ax_max = plt.subplots(figsize=(9, 9))
    min_results = []
    for i in range(subnum):
            m3n.append([])
            min_results.append([])
            for j in range(subnum):
                x_train = np.concatenate((w_subsets[i][0], b_subsets[j][0]), axis=0)
                y_train = np.concatenate((w_subsets[i][1], b_subsets[j][1]), axis=0)
                if submodule == 0:
                    mod = SVC(gamma=2, C=1, probability=True)
                elif submodule == 1:
                    mod = MLPClassifier(activation='logistic', alpha=1, learning_rate_init=lr, max_iter=8000)
                    
                m3n[i].append(mod.fit(x_train, y_train.ravel()))
                # plot decision boundary
                z = mod.predict_proba(np.c_[xx.ravel(), yy.ravel()])[:, 1] # probability that it belongs to class 1
                min_results[i].append(z.reshape(-1, 1))
                ax[i, j].contourf(xx, yy, (z >= 0.5).reshape(xx.shape), cmap=cm, alpha=0.6)
                # plot training points
                ax[i, j].scatter(x_train[:, 0], x_train[:, 1], s=5, c=y_train, cmap=cm_points)
                ax[i, j].set_xticks(())
                ax[i, j].set_yticks(())
                print('({}, {}) finished'.format(i, j))
    
    fig.tight_layout()
    fig.savefig('./m3n.png')

    # plot min step
    min_gate = []
    for i in range(subnum):
        r = np.concatenate(min_results[i], axis=1).min(axis=1)
        min_gate.append(r.reshape(-1, 1))
        ax_min[i].contourf(xx, yy, (r >= 0.5).reshape(xx.shape), cmap=cm, alpha=0.6)
        ax_min[i].scatter(coordinates[:, 0], coordinates[:, 1], s=5, c=labels, cmap=cm_points)
        ax_min[i].set_xticks(())
        ax_min[i].set_yticks(())
    
    fig_min.tight_layout()
    fig_min.savefig('./min_gate.png')

    # plot max step
    max_gate = np.concatenate(min_gate, axis=1).max(axis=1)
    ax_max.contourf(xx, yy, (max_gate >= 0.5).reshape(xx.shape), cmap=cm, alpha=0.6)
    ax_max.scatter(coordinates[:, 0], coordinates[:, 1], s=10, c=labels, cmap=cm_points)
    ax_max.set_xticks(())
    ax_max.set_yticks(())
    fig_max.savefig('./max_gate.png')

    return m3n


def predict(m3n, x, y, subnum):
    min_results = []
    for i in range(subnum):
        min_results.append([])
        for j in range(subnum):
            pred = m3n[i][j].predict_proba(x)[:, 1]
            min_results[i].append(pred.reshape(-1, 1))
    
    min_gate = []
    for i in range(subnum):
        r = np.concatenate(min_results[i], axis=1).min(axis=1)
        min_gate.append(r.reshape(-1, 1))
    
    max_gate = np.concatenate(min_gate, axis=1).max(axis=1)
    pred = (max_gate >= 0.5).astype(int)
    # pay attention to the shape of pred and y
    correct = (pred.ravel() == y.ravel()).sum()
    accuracy = correct / y.shape[0]
    return accuracy


txt_file = r'./two-spiral.txt'
save_path = r'.'

parser = argparse.ArgumentParser()
parser.add_argument('--submodule', type=int, default=0, help='0-RBF SVM, 1-Perceptron')
parser.add_argument('--subnum', type=int, default=3, help='The number of subsets that you would like decompose')
parser.add_argument('--lr', type=float, default=1e-2, help='Learning rate')
args = parser.parse_args()

coordinates, labels = parseTxt(txt_file)
coordinates, labels = np.array(coordinates, dtype=np.float), np.array(labels, dtype=np.float)
# coordinates = StandardScaler().fit_transform(coordinates)
# x_min, x_max = coordinates[:, 0].min() - 0.5, coordinates[:, 0].max() + 0.5
# y_min, y_max = coordinates[:, 1].min() - 0.5, coordinates[:, 1].max() + 0.5
num_samples = labels.shape[0]
num_white = labels.sum()
num_black = num_samples - num_white
print('Data size: ', num_samples) # 194
print('Number of white points(1): ', num_white) # 97
print('Number of black points(0): ', num_black) # 97

# boolean index should be 1 dimensional
w_indices = (labels == 1).ravel()
b_indices = (labels == 0).ravel()
w_coordinates, w_labels = coordinates[w_indices], labels[w_indices]
b_coordinates, b_labels = coordinates[b_indices], labels[b_indices]

w_subsets = divideData(w_coordinates, w_labels, args.subnum)
b_subsets = divideData(b_coordinates, b_labels, args.subnum)

m3n = trainM3Net(args.submodule, args.subnum, args.lr, w_subsets, b_subsets, coordinates, labels)

accuracy = predict(m3n, coordinates, labels, args.subnum)

print('Prediction accuracy: {:.4f}'.format(accuracy))

