import torch
from torch import nn



class CharRNN(nn.Module):
    """
    Character-level LSTM.
    
    Parameters
    ----------
    hidden_size:
        Number of output features for LSTM.
    dropout:
        Dropout probabilityfor LSTM.
    batch_size:
        Number of sequences in a batch.
    D:
        Number of directions: uni- or bidirectional architecture for LSTM.
    num_layers:
        Number of LSTM stacks.
    
    Returns
    -------
    output:
        Shape: [batch_size, sequence_length, num_features]
    hidden_state:
        Tuple containing:
        - Short-term hidden state
            Shape: [batch_size, sequence_length, num_features]
        - Cell state
            Shape: [batch_size, sequence_length, num_features]
    
    """
    def __init__(self, hidden_size = 128, dropout = 0.25,
                 batch_size = 32, D = 1, num_layers = 2):
        
        super(CharRNN, self).__init__()
        
        self.hidden_size = hidden_size
        self.dropout_rate = dropout
        self.num_layers = num_layers
        self.batch_size = batch_size
        self.D = D
        
        self.lstm = nn.LSTM(input_size = len(unique_chars), hidden_size = self.hidden_size,
                            dropout = self.dropout_rate, batch_first = True,
                            bidirectional = True if self.D == 2 else False, bias = True,
                            num_layers = self.num_layers)
        
        self.fc = nn.Linear(self.D*self.hidden_size, len(unique_chars))
        
    def forward(self, x, hidden_state):
        outputs, hidden_state = self.lstm(x, hidden_state)
        outputs = outputs.contiguous().view(-1, self.D*self.hidden_size)
        outputs = self.fc(outputs)
        
        return outputs, hidden_state
    
    def init_hidden_state(self, mean, stddev):
        """
        Initialize hidden state and context tensors.
        """
        
        h = torch.distributions.Normal(mean, stddev).sample((self.D*self.num_layers, self.batch_size, self.hidden_size))
        c = torch.distributions.Normal(mean, stddev).sample((self.D*self.num_layers, self.batch_size, self.hidden_size))
        
        return (h, c)
        