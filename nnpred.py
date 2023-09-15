import numpy as np
import datetime
import pandas as pd
from sklearn.preprocessing import StandardScaler
import torch
from torch import nn
import torch.nn.functional as F
from utils import MAPE, MAE




class LSTMREG(nn.Module):
    def __init__(self,input_size,hidden_size, output_size, num_layers=2):
        super(LSTMREG,self).__init__()
        self.rnn = nn.LSTM(input_size,hidden_size,num_layers) 
        self.reg = nn.Linear(hidden_size,output_size) 


    def forward(self,x):
        x = x.unsqueeze(2)
        rnnout, _ = self.rnn(x)

        rnnout = rnnout[:, -1, :]
    
        return self.reg(rnnout)


class GRUREG(nn.Module):
    def __init__(self,input_size,hidden_size, output_size,num_layers=2):
        super(GRUREG,self).__init__()
        self.rnn = nn.GRU(input_size,hidden_size,num_layers) 
        self.reg = nn.Linear(hidden_size,output_size) 


    def forward(self,x):   
        x = x.unsqueeze(2)
        rnnout, _ = self.rnn(x)

        rnnout = rnnout[:, -1, :]

        return self.reg(rnnout)


def evaluate_model(model, criterion, data_iter):
    model.eval()
    l_sum, n = 0.0, 0
    with torch.no_grad():
        for x, y in data_iter:
            y_pred = model(x)
            l = criterion(y_pred, y)
            l_sum += l.item() * y.shape[0]
            n += y.shape[0]
        return l_sum / n



def nn_train(data_x, data_y, model_name, learning_rate=1e-3, device='cuda:0'):

    torch.manual_seed(100)
    torch.cuda.manual_seed(100)

    assert len(data_x.shape) == 2, 'input shape not match'

    his_len = data_x.shape[1]
    pred_len = data_y.shape[1]

    data_x = torch.Tensor(data_x).to(device)
    data_y = torch.Tensor(data_y).to(device)

    # split validation set
    split_idx  = int(data_x.shape[0] * 0.8)
    train_x, train_y = data_x[:split_idx], data_y[:split_idx]
    val_x, val_y = data_x[split_idx:], data_y[split_idx:]

    train_set = torch.utils.data.TensorDataset(train_x, train_y)
    val_set = torch.utils.data.TensorDataset(val_x, val_y)
    train_iter = torch.utils.data.DataLoader(train_set, batch_size = 1024, shuffle=True)
    val_iter = torch.utils.data.DataLoader(val_set, batch_size = 1024, shuffle=False)
    
    

    if model_name == 'LSTM':
        model = LSTMREG(input_size = 1, hidden_size = 128, output_size = pred_len)
    
    elif model_name == 'GRU':
        model = GRUREG(input_size = 1, hidden_size = 128, output_size = pred_len)

    else:
        pass

    model = model.to(device)
    criterion = nn.L1Loss()
    optimizer = torch.optim.Adam(model.parameters(),lr=learning_rate)

    epoch_num = 500
    patience = 20
    wait = 0
    min_val_loss = np.inf

    for epoch in range(epoch_num):

        loss_sum, n = 0.0, 0
        model.train()
        for x, y in train_iter:
            optimizer.zero_grad()
            y_pred = model(x)
            loss = criterion(y_pred, y)
            loss.backward()
            optimizer.step()
            loss_sum += loss.item() * y.shape[0]
            n += y.shape[0]
        train_loss = loss_sum / n

        val_loss = evaluate_model(model, criterion, val_iter)
        if val_loss < min_val_loss:
            wait = 0
            min_val_loss = val_loss
        else:
            wait += 1
            if wait == patience:
                print('Early stopping at epoch: %d' % epoch)
                break

        print("epoch", epoch, 
              "train loss:", train_loss, "validation loss:", val_loss)

    # nn_pred(val_x, val_y, model)

    return model



def nn_pred(data_x, data_y, model, device='cuda:0'):

    data_x = torch.Tensor(data_x).to(device)
    data_y = torch.Tensor(data_y).to(device)

    test_set = torch.utils.data.TensorDataset(data_x, data_x)
    test_iter = torch.utils.data.DataLoader(test_set, batch_size = 1024, shuffle=False)


    pred_y = []

    model.eval()
    for x, _ in test_iter:
        pred = model(x)

        pred_y.append(pred.detach().cpu())
    
    pred_y = torch.vstack(pred_y)
    
    return pred_y.numpy()



def get_train_test (data, mode, his_len = 12, pred_len = 3, train_rate = 0.8):

    train_num = int(data.shape[0] * train_rate)
    XS, YS = [], []
    if mode == 'train':
        for i in range(train_num - pred_len - his_len + 1):
            x = data[i:i + his_len, :]
            y = data[i + his_len:i + his_len + pred_len, :]
            XS.append(x), YS.append(y)
    elif mode == 'test':
        for i in range(train_num - his_len,
                       data.shape[0] - pred_len - his_len + 1):
            x = data[i:i + his_len, :]
            y = data[i + his_len:i + his_len + pred_len, :]
            XS.append(x), YS.append(y)

    XS, YS = np.array(XS), np.array(YS)
    XS, YS = np.squeeze(XS), np.squeeze(YS)
    
    return XS, YS




