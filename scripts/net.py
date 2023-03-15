import torch
from torch import nn


class CharRNN(nn.Module):
    """
    Character-level LSTM.

    Parameters
    ----------
    input_size:
        Input (feature sze) for RNN.
    output_size:
        Input (feature sze) for RNN.
    hidden_size:
        Number of output features for RNN.
    dropout:
        Dropout probability for RNN.
    batch_size:
        Number of sequences in a batch.
    D:
        Number of directions: uni- or bidirectional architecture for RNN.
    num_layers:
        Number of RNN stacks.
    batch_size:
        Number of sequences in a batch.

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

    def __init__(
        self,
        input_size=32,
        hidden_size=128,
        dropout=0.25,
        batch_size=32,
        D=1,
        num_layers=2,
        output_size=32,
        base_rnn=nn.LSTM,
        device="cpu",
    ):
        super(CharRNN, self).__init__()

        self.base_rnn = base_rnn
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_size = hidden_size
        self.dropout_rate = dropout
        self.num_layers = num_layers
        self.batch_size = batch_size
        self.D = D

        self.device = device

        self.rnn = self.base_rnn(
            input_size=self.input_size,
            hidden_size=self.hidden_size,
            dropout=self.dropout_rate,
            batch_first=True,
            bidirectional=True if self.D == 2 else False,
            bias=True,
            num_layers=self.num_layers,
        )

        self.dropout = nn.Dropout(p=self.dropout_rate)
        self.fc = nn.Linear(self.D * self.hidden_size, self.output_size)

    def forward(self, x, hidden_state):
        outputs, hidden_state = self.rnn(x, hidden_state)
        outputs = self.dropout(outputs)
        outputs = outputs.contiguous().view(-1, self.D * self.hidden_size)
        outputs = self.fc(outputs)

        return outputs, hidden_state

    def init_hidden_state(self, mean=0, stddev=0):
        """
        Initialize hidden state and context tensors.
        """
        h = (
            torch.distributions.Normal(mean, stddev)
            .sample((self.D * self.num_layers, self.batch_size, self.hidden_size))
            .zero_()
        )
        h = h.to(self.device)

        if self.base_rnn == nn.LSTM:
            c = (
                torch.distributions.Normal(mean, stddev)
                .sample((self.D * self.num_layers, self.batch_size, self.hidden_size))
                .zero_()
            )
            c = c.to(self.device)
            h = (h, c)

        return h


def get_base_rnn(name):
    if name.lower() == 'lstm':
        base_rnn = nn.LSTM
    elif name.lower() == 'gru':
        base_rnn = nn.GRU
    else:
        base_rnn = nn.RNN

    return base_rnn
