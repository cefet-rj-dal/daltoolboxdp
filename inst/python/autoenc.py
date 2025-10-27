"""
PyTorch autoencoder used by daltoolboxdp via reticulate.

Entry points called from R (see R/autoenc_e.R and R/autoenc_ed.R):
  - autoenc_create(input_size, encoding_size) -> nn.Module
  - autoenc_fit(model, data, batch_size=32, num_epochs=1000, learning_rate=1e-3)
      Returns (model, train_loss_array, val_loss_array)
  - autoenc_encode(model, data, batch_size=32) -> np.ndarray [n_samples, encoding_size]
  - autoenc_encode_decode(model, data, batch_size=32) -> np.ndarray [n_samples, input_size]

Data expectations:
  - data: pandas.DataFrame with shape (n_samples, input_size), numeric columns only.
  - Encoding returns a dense NumPy array suitable to be consumed back in R.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
torch.set_grad_enabled(True)
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from random import sample


class Autoencoder_TS(Dataset):
    def __init__(self, num_samples, input_size):
        self.data = np.random.randn(num_samples, input_size)

    def __init__(self, data):
        self.data = data

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, index):
        return self.data[index], self.data[index]

class Autoencoder(nn.Module):
    def __init__(self, input_size, encoding_size):
        super(Autoencoder, self).__init__()

        self.encoder = nn.Sequential(
            nn.Linear(input_size, 64),
            nn.ReLU(True),
            nn.Linear(64, encoding_size))

        self.decoder = nn.Sequential(
            nn.Linear(encoding_size, 64),
            nn.ReLU(True),
            nn.Linear(64, input_size))

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x
    
def autoenc_create(input_size, encoding_size):
  """Factory called from R to create the PyTorch autoencoder model.

  Parameters
  ----------
  input_size : int
      Number of input features (must match columns in the R data.frame).
  encoding_size : int
      Size of the latent bottleneck.

  Returns
  -------
  torch.nn.Module
      Initialized autoencoder on CPU.
  """
  input_size = int(input_size)
  encoding_size = int(encoding_size)
  
  autoencoder = Autoencoder(input_size, encoding_size)
  autoencoder = autoencoder.float()
  return autoencoder  


def autoenc_train(autoencoder, data, batch_size=32, num_epochs = 1000, learning_rate = 0.001):
  """Internal training loop. Splits data each epoch and tracks losses.

  Returns (model, train_loss_np, val_loss_np) to match R expectations.
  """
  criterion = nn.MSELoss()
  optimizer = optim.Adam(autoencoder.parameters(), lr=learning_rate)

  train_loss = []
  val_loss = []
  
  for epoch in range(num_epochs):
      # Train Test Split
      array = data.to_numpy()
      array = array[:, :]
      
      val_sample = sample(range(1, data.shape[0], 1), k=int(data.shape[0]*0.3))
      train_sample = [v for v in range(1, data.shape[0], 1) if v not in val_sample]
      
      train_data = array[train_sample, :]
      val_data = array[val_sample, :]
      
      ds_train = Autoencoder_TS(train_data)
      ds_val = Autoencoder_TS(val_data)
      train_loader = DataLoader(ds_train, batch_size=batch_size)
      val_loader = DataLoader(ds_val, batch_size=batch_size)
    
      # Train
      train_epoch_loss = []
      val_epoch_loss = []
      autoencoder.train()
      for train_data in train_loader:
          train_input, _ = train_data
          train_input = train_input.float()
          optimizer.zero_grad()
          train_output = autoencoder(train_input)
          train_batch_loss = criterion(train_output, train_input)
          train_batch_loss.backward()
          optimizer.step()
          train_epoch_loss.append(train_batch_loss.item())
          
          
      # Validation
      autoencoder.eval()
      for val_data in val_loader:
          val_input, _ = val_data
          val_input = val_input.float()
          val_output = autoencoder(val_input)
          val_batch_loss = criterion(val_output, val_input)
          val_epoch_loss.append(val_batch_loss.item())
          
      train_loss.append(np.mean(train_epoch_loss))
      val_loss.append(np.mean(val_epoch_loss))
  
  return autoencoder, np.array(train_loss), np.array(val_loss)

def autoenc_fit(autoencoder, data, batch_size = 32, num_epochs = 1000, learning_rate = 0.001):
  """Entry point from R to fit the model.

  R receives a 3-tuple: (model, train_loss, val_loss) as a Python object.
  """
  batch_size = int(batch_size)
  num_epochs = int(num_epochs)
  
  autoencoder = autoenc_train(autoencoder, data, batch_size=batch_size, num_epochs = num_epochs, learning_rate = 0.001)
  return autoencoder


def encode_data(autoencoder, data_loader):
  # Helper: run encoder over a DataLoader and stack numpy batches
  encoded_data = []
  for data in data_loader:
      inputs, _ = data
      inputs = inputs.float()
      encoded = autoencoder.encoder(inputs)
      encoded_data.append(encoded.detach().numpy())

  encoded_data = np.concatenate(encoded_data, axis=0)

  return encoded_data

def autoenc_encode(autoencoder, data, batch_size = 32):
  """Entry point from R to obtain latent encodings.

  Returns
  -------
  np.ndarray of shape (n_samples, encoding_size)
  """
  array = data.to_numpy()
  array = array[:, :]
  
  ds = Autoencoder_TS(array)
  train_loader = DataLoader(ds, batch_size=batch_size)
  
  encoded_data = encode_data(autoencoder, train_loader)
  
  return(encoded_data)


def encode_decode_data(autoencoder, data_loader):
  # Helper: encode and then decode a full dataset, returning reconstructions
  encoded_decoded_data = []
  for data in data_loader:
      inputs, _ = data
      inputs = inputs.float()
      encoded = autoencoder.encoder(inputs)
      decoded = autoencoder.decoder(encoded)
      encoded_decoded_data.append(decoded.detach().numpy())

  encoded_decoded_data = np.concatenate(encoded_decoded_data, axis=0)

  return encoded_decoded_data


def autoenc_encode_decode(autoencoder, data, batch_size = 32):
  """Entry point from R to obtain reconstructions (decode(encode(x))).

  Returns
  -------
  np.ndarray of shape (n_samples, input_size)
  """
  batch_size = int(batch_size)
  
  array = data.to_numpy()
  array = array[:, :]
  
  ds = Autoencoder_TS(array)
  train_loader = DataLoader(ds, batch_size=batch_size)
  
  encoded_decoded_data = encode_decode_data(autoencoder, train_loader)
  
  return(encoded_decoded_data)
  
