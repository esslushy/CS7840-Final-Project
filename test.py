import numpy as np
import random
import torch
import torch.nn as nn
from entropy import mutual_info

# Binarization
def g(x):
    if x < 0:
        return 0
    else:
        return 1
    
# Gaussian Noise
def f(x):
    y =  100*np.cos(np.sum(x,axis=1)) - np.sum(x,axis=1)**2 + 5*np.random.normal(0,1, 10000)
    return np.array(list(map(g,y)))

# Discretization of continous values from the layers
def discretization(activations_list,bins):
    n_bins = bins

    bins = torch.linspace(torch.min(activations_list),
                        torch.max(activations_list), n_bins+1)
    activations_list = torch.bucketize(activations_list, bins)
            
    return activations_list

#Creating X and Y
x = np.array([[random.randint(0, 1) for i in range(10)] for i in range(10000)])
y = f(x).reshape(-1,1)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(x,y, test_size=0.3, random_state=42)

input_shape = len(X_train[0])
output_shape = len(y_train[0])

X_train, X_test, y_train, y_test = torch.tensor(X_train, dtype=torch.float32), torch.tensor(X_test, dtype=torch.float32), torch.tensor(y_train, dtype=torch.float32), torch.tensor(y_test, dtype=torch.float32)

from torch.utils.data import TensorDataset, DataLoader
train_dataset = TensorDataset(X_train, y_train)
train_loader = DataLoader(train_dataset, batch_size=160, shuffle=True)

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear_1 = nn.Linear(input_shape, 3)
        self.linear_2 = nn.Linear(3, 2)
        self.linear_3 = nn.Linear(2, output_shape)

    def forward(self, x):
        step1 = nn.functional.tanh(self.linear_1(x))
        step2 = nn.functional.tanh(self.linear_2(step1))
        step3 = nn.functional.sigmoid(self.linear_3(step2))
        return step1, step2, step3

net = Net()
optim = torch.optim.SGD(net.parameters(), lr=0.01)

import matplotlib.pyplot as plt


EPOCHS = 100

I_XT = np.zeros((3, EPOCHS))
I_TY = np.zeros((3, EPOCHS))

for i in range(EPOCHS):
    for x, y in train_loader:
        optim.zero_grad()
        layer1, layer2, layer3 = net(x)
        loss = nn.functional.binary_cross_entropy(layer3, y)
        print(loss.item())
        loss.backward()
        optim.step()

    # Info calc
    layer1, layer2, layer3 = net(X_train)
    layer1 = discretization(layer1, 30)
    layer2 = discretization(layer2, 30)
    layer3 = discretization(layer3, 30)

    I_XT[0, i] = mutual_info(layer1, X_train)
    I_XT[1, i] = mutual_info(layer2, X_train)
    I_XT[2, i] = mutual_info(layer3, X_train)
    I_TY[0, i] = mutual_info(layer1, y_train)
    I_TY[1, i] = mutual_info(layer2, y_train)
    I_TY[2, i] = mutual_info(layer3, y_train)


def plot_information_plane(IXT_array, ITY_array, num_epochs, every_n, I_XY):
    assert len(IXT_array) == len(ITY_array)

    max_index = len(IXT_array)

    plt.figure(figsize=(12, 6),dpi=150)
    plt.xlabel(r'$I(X;T)$')
    plt.ylabel(r'$I(T;Y)$')

    cmap = plt.get_cmap('gnuplot')
    colors = [cmap(i) for i in np.linspace(0, 1, num_epochs)]
    cmap_layer = plt.get_cmap('Greys')
    clayer = [cmap_layer(i) for i in np.linspace(0, 1, max_index)]

    for i in range(0, max_index):
        IXT = IXT_array[i, :]
        ITY = ITY_array[i, :]

        plt.plot(IXT,ITY,color=clayer[i],linestyle=None,linewidth=2,label='Layer {}'.format(str(i)))
        plt.scatter(IXT,ITY,marker='o',c=colors,s=200,alpha=1)#,zorder=10)

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=0, vmax=1))
    sm._A = []
    # cbar = plt.colorbar(sm, ticks=[])
    # cbar.set_label('Num epochs')
    # cbar.ax.text(0.5, -0.01, 0, transform=cbar.ax.transAxes, va='top', ha='center')
    # cbar.ax.text(0.5, 1.0, str(num_epochs), transform=cbar.ax.transAxes, va='bottom', ha='center')
    plt.axhline(y = I_XY, color = 'red', linestyle = ':', label=r'$I[X,Y]$')
    plt.legend()
    plt.show()


plot_information_plane(I_XT,I_TY,EPOCHS,1, mutual_info(X_train,y_train))
