import torch.nn as nn


class RNNModel(nn.Module):

    def __init__(self, n_token, n_inp, n_hidden, n_layers, dropout=0.5):
        super(RNNModel, self).__init__()
        self.drop = nn.Dropout(dropout)
        self.encoder = nn.Embedding(n_token, n_inp) # Token2Embeddings
        self.rnn = nn.LSTM(n_inp, n_hidden, n_layers, dropout=dropout) #(seq_len, batch_size, emb_size)
        self.decoder = nn.Linear(n_hidden, n_token)
        self.initial_weights()
        self.n_hidden = n_hidden
        self.n_layers = n_layers

    def initial_weights(self):
        initrange = 0.05
        self.encoder.weight.data.uniform_(-initrange, initrange)
        self.decoder.bias.data.fill_(0)
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, input, hidden):
        emb = self.drop(self.encoder(input))
        output, hidden = self.rnn(emb, hidden)
        output = self.drop(output)
        decoded = self.decoder(output.view(output.size(0)*output.size(1), output.size(2)))
        return decoded, hidden

    def init_hidden(self, batchsize):
        weight = next(self.parameters()).data
        return weight.new_zeros(self.n_layers, batchsize, self.n_hidden), weight.new_zeros(self.n_layers, batchsize, self.n_hidden)