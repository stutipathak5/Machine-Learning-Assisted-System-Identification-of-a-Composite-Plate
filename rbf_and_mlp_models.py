import torch
import numpy as np
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
import matplotlib.pyplot as plt

"DEFINING DATASET"

class MyDataset(Dataset):

    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return self.X.size(0)

    def __getitem__(self, idx):
        X = self.X[idx]
        y = self.y[idx]
        return (X, y)

"DEFINING RBF_CLUST"

class RBF_CLUST(nn.Module):

    def __init__(self, in_features, out_features, basis_func):
    
        super().__init__()
        self.in_features = in_features                                         
        self.out_features = out_features                                      
        self.centres = torch.from_numpy(c.astype(np.float32))
        self.sigmas = torch.from_numpy(r.astype(np.float32))
        self.basis_func = basis_func


    def forward(self, input):
    
        size = (input.size(0), self.out_features, self.in_features)            
        x = input.unsqueeze(1).expand(size)                                     
        c = self.centres.unsqueeze(0).expand(size)                                  
        distances = (x - c).pow(2).sum(-1).pow(0.5) * self.sigmas.unsqueeze(0).pow(-1)  
        
        return self.basis_func(distances) 

"DEFINING RBF FUNCTIONS"

def gaussian(alpha):
  phi = torch.exp(-1 * alpha.pow(2))
  return phi

def multiquadric(alpha):
  phi = (alpha ** 2 + 1) ** (1 / 2)
  return phi

def inverse_multiquadric(alpha):
  phi = (alpha ** 2 + 1) ** (-1 / 2)
  return phi

def cauchy(alpha):
  phi = (alpha ** 2 + 1) ** (-1)
  return phi

"DEFINING ONE_LAYER_RBF_NN"

class ONE_LAYER_RBF_NN(nn.Module):

  def __init__(self, no_RBF_neurons, no_in_features, RBF_function):

    super().__init__()
    self.no_RBF_neurons = no_RBF_neurons
    self.no_in_features = no_in_features
    self.RBF_function = RBF_function
    self.RBF = RBF_CLUST(no_in_features, no_RBF_neurons, RBF_function)
    self.Linear = nn.Linear(no_RBF_neurons, 1)

  def forward(self, x):
    x = self.RBF(x)
    return self.Linear(x)

  def fit(self, x, y, epochs, batch_size, lr, loss_func, train_size, test_size):

    self.train()
    trainset, testset = torch.utils.data.random_split(MyDataset(x, y), (train_size, test_size))
    trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True)
    testloader = DataLoader(testset, batch_size=test_size, shuffle=False)
    
    optimiser = torch.optim.Adam(self.parameters(), lr=lr)
    epoch = 0
    loss_vals = []
    test_loss = []
    
    while epoch < epochs:
      epoch += 1
      batches = 0
      epoch_loss = []
      
      for x_batch, y_batch in trainloader:
        batches += 1
        optimiser.zero_grad()
        y_hat = self.forward(x_batch)
        loss = loss_func(y_hat, y_batch.unsqueeze(1))
        loss.backward()
        optimiser.step()
        epoch_loss.append(loss.item())
        
      loss_vals.append(sum(epoch_loss) / len(epoch_loss))
      with torch.no_grad():
        for x_test, y_test in testloader:
          epoch_test_loss = loss_func(self.forward(x_test), y_test.unsqueeze(1))
          test_loss.append(epoch_test_loss.item())

    plt.figure(figsize=(8, 5))
    plt.xlabel('Number of epochs', fontsize=20)
    plt.ylabel('Loss', fontsize=20)
    plt.plot(np.linspace(1, epochs, epochs).astype(int), loss_vals, 'o-', markersize=4, linewidth=2, c="tomato", label="Training Loss")
    plt.plot(np.linspace(1, epochs, epochs).astype(int), test_loss, 'o-', markersize=4, linewidth=2, c="cornflowerblue", label="Testing Loss")
    plt.title("Two Layer RBF", fontsize=20)
    plt.legend(loc="upper right", fontsize=20)
    plt.grid()
    plt.show()
    
    print("loss_vals=", loss_vals)
    print("test_loss=", test_loss)
    print("min train loss: ", min(loss_vals))
    print("min test loss: ", min(test_loss))
    print("epoch for min test loss: ", np.argmin(test_loss) + 1)
    print("train loss for min test loss: ", loss_vals[np.argmin(test_loss)])
    print("final test loss: ", test_loss[-1])

"FUNCTION FOR RADIAL BASIS FUNCTION NEURAL NETWORK SOLVED USING LEAST SQUARES METHOD"

def RBF_LS4(x, y, sc, train_size, dev_size, test_size):
  
  xtr = x[0:train_size,:]
  ytr = y[0:train_size,:]
  xdev = x[train_size:train_size+dev_size,:]
  ydev = y[train_size:train_size+dev_size,:]
  xte = x[train_size+dev_size:train_size+dev_size+test_size,:]
  yte = y[train_size+dev_size:train_size+dev_size+test_size,:]
  
  no_tr_sam, in_features= xtr.size()
  no_dev_sam = xdev.size()[0]
  no_te_sam = xte.size()[0]
  
  size1 = (no_tr_sam, no_tr_sam, in_features)         
  x1 = xtr.unsqueeze(1).expand(size1)                                
  c1 = xtr.unsqueeze(0).expand(size1)
  s = (sc/np.sqrt(-np.log(0.5)))*torch.ones(no_tr_sam)                                
  z1 = (x1 - c1).pow(2).sum(-1).pow(0.5)*s.unsqueeze(0).pow(-1)
  a1 = gaussian(z1)
  w, _ = torch.lstsq(ytr,torch.cat((a1,\
         torch.ones((no_tr_sam,1))), 1))
         
  tr_pred = torch.matmul(torch.cat((a1,\
            torch.ones((no_tr_sam,1))), 1),w)
  rmse_tr = torch.sqrt(sum((ytr-tr_pred.squeeze(1))**2))/no_tr_sam
  
  size2 = (no_dev_sam, no_tr_sam, in_features) 
  x2 = xdev.unsqueeze(1).expand(size2)
  c2 = xtr.unsqueeze(0).expand(size2)
  z2 = (x2 - c2).pow(2).sum(-1).pow(0.5)*s.unsqueeze(0).pow(-1)
  a2 = gaussian(z2)
  
  dev_pred = torch.matmul(torch.cat((a2,\
            torch.ones((no_dev_sam,1))), 1),w)
  rmse_dev = torch.sqrt(sum((ydev-dev_pred.squeeze(1))**2))/no_dev_sam
  
  size3 = (no_te_sam, no_tr_sam, in_features) 
  x3 = xte.unsqueeze(1).expand(size3)
  c3 = xtr.unsqueeze(0).expand(size3)
  z3 = (x3 - c3).pow(2).sum(-1).pow(0.5)*s.unsqueeze(0).pow(-1)
  a3 = gaussian(z3)
  
  te_pred = torch.matmul(torch.cat((a3,\
            torch.ones((no_te_sam,1))), 1),w)
            
  return te_pred, rmse_tr, rmse_dev
  
"DEFINING TWO_LAYER_MLP"

class TWO_LAYER_MLP(nn.Module):

    def __init__(self, input_size, hidden_size, hidden_size1):
    
            super().__init__()
            self.input_size = input_size
            self.hidden_size  = hidden_size
            self.hidden_size1  = hidden_size1
            self.fc1 = nn.Linear(self.input_size, self.hidden_size)
            self.relu = nn.ReLU()
            self.fc2 = nn.Linear(self.hidden_size, self.hidden_size1)
            self.relu1 = nn.ReLU()
            self.fc3 = nn.Linear(self.hidden_size1, 1)

    def forward(self, x):
      
            x = self.fc1(x)
            x = self.relu(x)
            x = self.fc2(x)
            x = self.relu1(x)
            
            return self.fc3(x)

    def fit(self, x, y, epochs, batch_size, lr, loss_func, train_size, test_size):
      
        self.train()
        trainset, testset = torch.utils.data.random_split(MyDataset(x, y), (train_size, test_size))
        trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True)
        testloader = DataLoader(testset, batch_size = test_size, shuffle = False)
        
        optimiser = torch.optim.Adam(self.parameters(), lr=lr, weight_decay=1e-5)
        epoch = 0
        loss_vals=[]
        test_loss = []
        
        while epoch < epochs:
            epoch += 1
            batches = 0
            epoch_loss = []
            
            for x_batch, y_batch in trainloader:
                batches += 1
                optimiser.zero_grad()
                y_hat = self.forward(x_batch)
                loss = loss_func(y_hat, y_batch.unsqueeze(1))
                loss.backward()
                optimiser.step()
                epoch_loss.append(loss.item())
                
            loss_vals.append(sum(epoch_loss)/len(epoch_loss))   
            with torch.no_grad():
                for x_test, y_test in testloader:
                    epoch_test_loss = loss_func(self.forward(x_test), y_test.unsqueeze(1))
                    test_loss.append(epoch_test_loss.item())
                    
        plt.figure(figsize=(8,5))
        plt.xlabel('Number of epochs', fontsize=20)
        plt.ylabel('Loss', fontsize=20)  
        plt.plot(np.linspace(1, epochs, epochs).astype(int), loss_vals, 'o-', markersize=4, linewidth=2, c="dodgerblue", label="Training Loss")
        plt.plot(np.linspace(1, epochs, epochs).astype(int), test_loss, 'o-', markersize=4, linewidth=2, c="gold",label="Testing Loss")
        plt.title("Three Layer MLP", fontsize=20)
        plt.legend(loc="upper right", fontsize=20)
        plt.grid()
        plt.show()
        
        print("loss_vals=", loss_vals)
        print("test_loss=", test_loss)
        print("min train loss: ", min(loss_vals))
        print("min test loss: ", min(test_loss))
        print("epoch for min test loss: ", np.argmin(test_loss)+1)
        print("train loss for min test loss: ", loss_vals[np.argmin(test_loss)])
        print("final test loss: ", test_loss[-1])
