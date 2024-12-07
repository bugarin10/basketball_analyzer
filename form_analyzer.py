import torch.nn as nn


class BallAnalyzer(nn.Module):
    def __init__(
        self, input_dim, hidden_dim, output_dim, n_layers, dropout_rate, bidirectional
    ):
        super(BallAnalyzer, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.n_layers = n_layers
        self.dropout_rate = dropout_rate
        self.bidirectional = bidirectional
        self.lstm = nn.LSTM(
            input_dim,
            hidden_dim,
            n_layers,
            dropout=dropout_rate,
            bidirectional=bidirectional,
        )
        self.linear = nn.Linear(hidden_dim, output_dim)
        pass

    def forward(self, x):
        out, _ = self.lstm(x)
        out = out[:, -1, :]
        out = self.fc(out)
        return out
