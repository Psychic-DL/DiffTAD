import torch
import torch.nn as nn

class Seq2Seq_LSTM(nn.Module):
    def __init__(self, input_size, hidden_size,
                 num_layers=1, dropout=0, batch_first=True):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.batch_first = batch_first

        seq_lstm = [nn.LSTM(input_size, hidden_size)]

        for _ in range(num_layers - 1):
            seq_lstm.append(nn.LSTM(hidden_size, hidden_size))

        self.seq_lstm = nn.Sequential(*seq_lstm)
        self.drop = nn.Dropout(dropout)
        self.depth_lstm = nn.LSTM(hidden_size, hidden_size)
        self.layernorms = nn.ModuleList([torch.nn.LayerNorm(hidden_size) for _ in range(num_layers)])

    def forward(self, input, hidden=None):
        if self.batch_first:
            input = input.transpose(0, 1)
        time_output = input
        time_results = []
        if hidden is None:
            hidden = [None for _ in self.seq_lstm]
        else:
            all_h, all_c = hidden
            hidden = [(h.unsqueeze(0), c.unsqueeze(0))
                      for h, c in zip(all_h, all_c)]
        next_hidden = []
        next_cell = []
        for lstm, state, layernorm in zip(self.seq_lstm, hidden, self.layernorms):
            # seq x bs x hidden
            time_output = layernorm(time_output)
            time_output, (next_h, next_c) = lstm(time_output, state)
            next_hidden.append(next_h)
            next_cell.append(next_c)
            time_output = self.drop(time_output)
            time_results.append(time_output)

        time_results = torch.stack(time_results)  # depth x seq x bs x hidden
        depth, seq, bs, hidden = time_results.size()
        _, (depth_h, depth_c) = self.depth_lstm(time_results.view(depth, seq * bs, hidden))
        output = depth_c  # seq*bs x hidden
        output = output.view(seq, bs, hidden) + time_output
        next_state = (torch.stack(next_hidden[::-1]).squeeze(1),
                      torch.stack(next_cell[::-1]).squeeze(1))

        if self.batch_first:
            output = output.transpose(0, 1)

        return output, next_state
