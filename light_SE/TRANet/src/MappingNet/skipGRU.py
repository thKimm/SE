import torch
import torch.nn as nn
from torch.nn.init import xavier_normal_
from torch.autograd import Function
    
class STEFunction(Function):
    @staticmethod
    def forward(cls,x):
        return x.round()

    @staticmethod
    def backward(cls,grad):
        return grad


class STELayer(nn.Module):
    def __init__(self):
        super(STELayer, self).__init__()

    def forward(self, x):
        binarizer = STEFunction.apply
        return binarizer(x)


class SkipGRUCell(nn.Module):
    def __init__(self, in_channels, hidden_size):
        super(SkipGRUCell, self).__init__()
        self.ste = STELayer()
        self.cell = nn.GRUCell(in_channels, hidden_size)
        self.linear = nn.Linear(hidden_size, 1)

        xavier_normal_(self.linear.weight)
        self.linear.bias.data.fill_(1)

    def forward(self, x, u, h, skip=False, delta_u=None):
        # x: (bs, in_channels)
        # u: (bs, 1)
        # h: (bs, hidden_size)
        # skip: [False or True] * bs
        # delta_u: [skip=True -> (1) / skip=False -> None] * bs

        bs = x.shape[0]
        binarized_u = self.ste(u)                # (bs, 1)

        skip_idx = [i for i, cur_skip in enumerate(skip) if cur_skip]
        skip_num = len(skip_idx)
        no_skip = [not cur_skip for cur_skip in skip]

        if skip_num > 0:
            # (skip_num, in_channels), (skip_num, 1), (skip_num, hidden_size)
            x_s, u_s, h_s = x[skip], u[skip], h[skip]
            binarized_u_s = binarized_u[skip]        # (skip_num, 1)

            # (skip_num, 1)
            delta_u_s = [cur_delta_u for cur_skip,
                         cur_delta_u in zip(skip, delta_u) if cur_skip]
            delta_u_s = torch.stack(delta_u_s)

            # computing skipped parts
            new_h_s = h_s * (1 - binarized_u_s)        # (skip_num, hidden_size)
            new_u_s = torch.clamp(u_s + delta_u_s, 0, 1) * \
                (1 - binarized_u_s)  # (skip_num, 1)

        if skip_num < bs:
            # (bs-skip_num, in_channels), (bs-skip_num, 1), (bs-skip_num, hidden_size)
            x_n, u_n, h_n = x[no_skip], u[no_skip], h[no_skip]
            binarized_u_n = binarized_u[no_skip]  # (bs-skip_num, 1)

            # computing non-skipped parts
            new_h_n = self.cell(x_n, h_n)  # (bs-skip_num, hidden_size)
            new_h_n = new_h_n * binarized_u_n            # (bs-skip_num, hidden_size)
            delta_u_n = torch.sigmoid(self.linear(new_h_n))        # (bs-skip_num, 1)
            new_u_n = delta_u_n * binarized_u_n                    # (bs-skip_num, 1)

        # merging skipped and non-skipped parts back
        if 0 < skip_num < bs:
            idx = torch.full((bs,), -1, dtype=torch.long)
            idx[skip_idx] = torch.arange(0, len(skip_idx), dtype=torch.long)
            idx[idx==-1] = torch.arange(len(skip_idx), bs, dtype=torch.long)

            new_u = torch.cat([new_u_s, new_u_n], 0)[idx]        # (bs, 1)
            new_h = torch.cat([new_h_s, new_h_n], 0)[idx]        # (bs, hidden_size)
            delta_u = torch.cat([delta_u_s, delta_u_n], 0)[idx]    # (bs, 1)

        # no need to merge when skip doesn't exist
        elif skip_num == 0:
            new_u = new_u_n
            new_h = new_h_n
            delta_u = delta_u_n

        # no need to merge when everything is skip
        elif skip_num == bs:
            new_u = new_u_s
            new_h = new_h_s
            delta_u = delta_u_s

        n_skips_after = (0.5 / new_u).ceil() - 1  # (bs, 1)
        return binarized_u, new_u, (new_h,), delta_u, n_skips_after

class SkipGRU(nn.Module):
    def __init__(self, in_channels, hidden_size, layer_num=2, return_total_u=False, learn_init=False, batch_first = False):
        super(SkipGRU, self).__init__()
        self.in_channels = in_channels
        self.hidden_size = hidden_size
        self.layer_num = layer_num
        self.return_total_u = return_total_u
        self.batch_first = batch_first

        cur_cell = SkipGRUCell

        self.cells = nn.ModuleList([cur_cell(in_channels, hidden_size)])
        for _ in range(self.layer_num - 1):
            cell = cur_cell(hidden_size, hidden_size)
            self.cells.append(cell)

        self.hiddens = self.init_hiddens(learn_init)

    def init_hiddens(self, learn_init):
        if learn_init:
            h = nn.Parameter(torch.randn(self.layer_num, 1, self.hidden_size))
        else:
            h = nn.Parameter(torch.zeros(self.layer_num, 1, self.hidden_size), requires_grad=False)
        return h

    def forward(self, x, hiddens=None):
        device = x.device
        if self.batch_first : 
            x = torch.permute(x,(1,0,2))
        
        x_len, bs, _ = x.shape    # (x_len, bs, in_channels)

        if hiddens is None:
            h = self.hiddens
        else:
            h = hiddens
            h = h.repeat(1, bs, 1)
        u = torch.ones(self.layer_num, bs, 1).to(device)            # (l, bs, 1)

        hs = []
        lstm_input = x             # (x_len, bs, in_channels)

        skip = [False] * bs
        delta_u = [None] * bs

        binarized_us = []

        for i in range(self.layer_num):
            cur_hs = []
            cur_h = h[i].unsqueeze(0)  # (1, bs, hidden_size)
            cur_u = u[i]               # (bs, 1)

            for j in range(x_len):
                # (bs, 1), ((bs, hidden_size), (bs, hidden_size)), (bs, 1), (bs, 1)
                binarized_u, cur_u, cur_h, delta_u, n_skips_after = self.cells[i](
                    lstm_input[j], cur_u, cur_h[0], skip, delta_u)
                binarized_us.append(binarized_u)
                skip = (n_skips_after[:, 0] > 0).tolist()

                # (1, bs, hidden_size) / (1, bs, hidden_size)
                cur_h = cur_h[0].unsqueeze(0)
                cur_hs.append(cur_h)

            # (x_len, bs, hidden_size)
            lstm_input = torch.cat(cur_hs, dim=0)
            hs.append(cur_h)

        # (bs, seq * layer_num)
        total_u = torch.cat(binarized_us, 1)
        # (x_len, bs, hidden_size)
        out = lstm_input
        # (l, bs, hidden_size)
        hs = torch.cat(hs, dim=0)
        
        if self.batch_first : 
            out = torch.permute(out,(1,0,2))

        if self.return_total_u:
            return out, (hs,), total_u
        return out, (hs,)