import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import TensorDataset
import torch.nn.functional as F
import sys

class TsLSTMNet(nn.Module):
  def __init__(self, n_neurons, input_shape):
    super(TsLSTMNet, self).__init__()
    self.lstm = nn.LSTM(input_size=input_shape, hidden_size=n_neurons)
    self.fc = nn.Linear(n_neurons, 1)
  
  def forward(self, x):
    out, _ = self.lstm(x)
    out = self.fc(out)
    return out


def ts_lstm_create(n_neurons, look_back):
  n_neurons = int(n_neurons)
  look_back = int(look_back)
  device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
  model = TsLSTMNet(n_neurons, look_back).to(device)
  return model    


def ts_lstm_train(epochs, lr, model, train_loader, opt_func=torch.optim.SGD):
  # to track the training loss as the model trains
  
  train_losses = []
  # to track the average training loss per epoch as the model trains
  avg_train_losses = []
  
  criterion = nn.MSELoss()
  last_error = sys.float_info.max
  last_epoch = 0
  
  convergency = epochs/10
  if (convergency < 100):
    convergency = 100
  
  optimizer = opt_func(model.parameters(), lr)
  for epoch in range(epochs):
    # train the model #
    model.train() # prep model for training
    for data, target in train_loader:
      # clear the gradients of all optimized variables
      model.zero_grad()
      # forward pass: compute predicted outputs by passing inputs to the model
      device = next(model.parameters()).device
      output = model(data.float().to(device))
      target = target.to(device)
      
      # calculate the loss
      loss = criterion(output, target.float())
      
      # backward pass: compute gradient of the loss with respect to model parameters
      loss.backward()
      # perform a single optimization step (parameter update)
      optimizer.step()
      # record training loss
      train_losses.append(loss.item())
    
    # validate the model #
    model.eval() # prep model for evaluation
    
    # calculate average loss over an epoch
    train_loss = np.average(train_losses)
    avg_train_losses.append(train_loss)
    
    if ((last_error - train_loss) > 0.001):
      last_error = train_loss
      last_epoch = epoch

    # clear lists to track next epoch
    train_losses = []
    
    if (train_loss == 0):
      break
        
    if (epoch - last_epoch > convergency):
      break

  return model, avg_train_losses


def ts_lstm_fit(model, df_train, n_epochs = 10000, lr = 0.001):
  n_epochs = int(n_epochs)
  
  X_train = df_train.drop('t0', axis=1).to_numpy()
  y_train = df_train.t0.to_numpy()
  X_train = X_train[:, :, np.newaxis]
  y_train = y_train[:, np.newaxis]	
  train_x = torch.from_numpy(X_train)
  train_y = torch.from_numpy(y_train)
  train_x = torch.permute(train_x, (2, 0, 1))
  train_labels = torch.permute(train_y, (1, 0))
  train_labels = train_labels[:, :, None]
  train_ds = TensorDataset(train_x, train_labels)
  
  BATCH_SIZE = 8
  train_loader = torch.utils.data.DataLoader(train_ds, batch_size = BATCH_SIZE, shuffle = False)
  
  model = model.float()
  model, train_loss = ts_lstm_train(n_epochs, lr, model, train_loader, opt_func=torch.optim.Adam)
  
  return model


def ts_lstm_predict(model, df_test):
  X_test = df_test.drop('t0', axis=1).to_numpy()
  y_test = df_test.t0.to_numpy()
  X_test = X_test[:, :, np.newaxis]
  y_test = y_test[:, np.newaxis]
  test_x = torch.from_numpy(X_test)
  test_labels = torch.from_numpy(y_test)
  test_x = torch.permute(test_x, (2, 0, 1))	
  test_labels = torch.permute(test_labels, (1, 0))
  test_labels = test_labels[:, :, None]
  test_ds = TensorDataset(test_x, test_labels)
  
  BATCH_SIZE = 8
  test_loader = torch.utils.data.DataLoader(test_ds, batch_size = BATCH_SIZE, shuffle = False)
  
  outputs = []
  with torch.no_grad():
    for xb, yb in test_loader:
      device = next(model.parameters()).device
      output = model(xb.float().to(device))      
      
      outputs.append(output.flatten())

  test_predictions = torch.vstack(outputs).squeeze(1)  
  test_predictions = test_predictions.flatten().cpu().numpy()

  return test_predictions
