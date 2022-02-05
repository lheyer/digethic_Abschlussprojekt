# LSTM model
import torch

class GeneralLSTM(torch.nn.Module):
  """
  input_size: The number of expected features in the input x
  num_hidden_units: The number hidden units in the LSTM-Layers
  batch_size: x.size(0) number of instances in the batch
  num_layers: for stacked LSTM-Layers num_layers >1 
  """

  def __init__(self, input_size, num_hidden_units, batch_size, num_layers=1):
    """
    Args:
        input_size (int): The number of expected features in the input x
        num_hidden_units (int): The number hidden units in the LSTM-Layers
        batch_size (int): x.size(0) number of instances in the batch
        num_layers (int): for stacked LSTM-Layers num_layers >1 
    """
    super(GeneralLSTM, self).__init__()


    self.input_size = input_size
    self.num_layers = num_layers
    self.num_hidden_units = num_hidden_units
    self.batch_size = batch_size

    self.lstm = torch.nn.LSTM(self.input_size, self.num_hidden_units, self.num_layers,batch_first=True)
    # -> batch_size, sequence_len, feature_len/input size
    self.fc = torch.nn.Linear(self.num_hidden_units,1)

    self.hidden = self.init_hidden(batch_size)

    
  def init_hidden(self,batch_size):
    h0 = torch.zeros(self.num_layers, batch_size, self.num_hidden_units).to(device)
    c0 = torch.zeros(self.num_layers, batch_size, self.num_hidden_units).to(device)
    
    return (c0,h0)


  def forward(self,x):
    out, h_state = self.lstm(x, self.hidden)
    self.hidden = h_state
    out = self.fc(out)

    return out, h_state
  
  
