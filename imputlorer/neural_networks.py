from torch import nn


class MLP(nn.Module):
    def __init__(self,
                 in_features=1,
                 out_features=1,
                 hidden_layers=[32, 32]):
        super().__init__()
        self.n_layers = len(hidden_layers)
        self.hidden_layers = hidden_layers

        self.fc_in = nn.Linear(in_features, self.hidden_layers[0])

        self.layers = nn.ModuleList()
        for i in range(self.n_layers-1):
            self.layers.append(nn.Linear(self.hidden_layers[i],
                                         self.hidden_layers[i-1]))

        self.fc_out = nn.Linear(self.hidden_layers[-1], out_features)

        self.relu = nn.ReLU()

        self.initialize()

    def initialize(self):
        nn.init.xavier_uniform_(self.fc_in.weight)
        nn.init.xavier_uniform_(self.fc_out.weight)
        for layer in self.layers:
            nn.init.xavier_uniform_(layer.weight)

    def forward(self, x):
        # b, m, n = x.shape
        # x = x.reshape(b, m, n)
        out = self.relu(self.fc_in(x))
        for layer in self.layers:
            out = self.relu(layer(out))
        out = self.fc_out(out)
        return out


class LSTM(nn.Module):
    def __init__(self,
                 input_size=1,
                 output_size=1,
                 hidden_size=16,
                 n_layers=1,
                 dropout=0.1):
        super().__init__()
        self.hidden_size = hidden_size
        self.dropout = dropout
        if n_layers == 1:
            self.dropout = 0.0

        self.lstm = nn.LSTM(input_size=input_size,
                            hidden_size=hidden_size,
                            num_layers=n_layers,
                            batch_first=True,
                            dropout=self.dropout)
        self.fc_out = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.fc_out(out)
        return out
