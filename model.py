import torch
from torch import nn
from torch.nn import functional as F
from quaternion_layers import QuaternionConv
from quaternion_ops import q_normalize

class InfoNCE(nn.Module):
    def __init__(self, temperature=0.01):
        super().__init__()
        self.temperature = temperature

    def forward(self, values, labels):
        return self.info_nce(values, labels)

    def normalize(self, x):
        return F.normalize(x, dim=-1)

    def info_nce(self, values, labels):
        logits = self.normalize(values)
        logits = logits / self.temperature
        logits = torch.softmax(logits, dim=-1)
        return F.binary_cross_entropy(logits, labels)


def get_param(shape):
    param = nn.Parameter(torch.Tensor(*shape))
    nn.init.xavier_normal_(param.data)
    return param


class LTEModel(nn.Module):
    def __init__(self, num_ents, num_rels, params=None):
        super(LTEModel, self).__init__()

        # self.bceloss = torch.nn.BCELoss()

        self.p = params
        self.bceloss = InfoNCE(self.p.temperature)
        self.init_embed = get_param((num_ents, self.p.init_dim))
        self.device = "cuda"

        self.init_rel = get_param((num_rels * 2, self.p.init_dim))

        self.bias = nn.Parameter(torch.zeros(num_ents))

        self.h_ops_dict = nn.ModuleDict({
            'p': nn.Linear(self.p.init_dim, self.p.gcn_dim, bias=False),
            'b': nn.BatchNorm1d(self.p.gcn_dim),
            'd': nn.Dropout(self.p.hid_drop),
            'a': nn.Tanh(),
        })

        self.t_ops_dict = nn.ModuleDict({
            'p': nn.Linear(self.p.init_dim, self.p.gcn_dim, bias=False),
            'b': nn.BatchNorm1d(self.p.gcn_dim),
            'd': nn.Dropout(self.p.hid_drop),
            'a': nn.Tanh(),
        })

        self.r_ops_dict = nn.ModuleDict({
            'p': nn.Linear(self.p.init_dim, self.p.gcn_dim, bias=False),
            'b': nn.BatchNorm1d(self.p.gcn_dim),
            'd': nn.Dropout(self.p.hid_drop),
            'a': nn.Tanh(),
        })

        self.x_ops = self.p.x_ops
        self.r_ops = self.p.r_ops
        self.diff_ht = False

    def calc_loss(self, pred, label):
        return self.loss(pred, label)

    def loss(self, pred, true_label):
        return self.bceloss(pred, true_label)

    def exop(self, x, r, x_ops=None, r_ops=None, diff_ht=False):
        x_head = x_tail = x
        if len(x_ops) > 0:
            for x_op in x_ops.split("."):
                if diff_ht:
                    x_head = self.h_ops_dict[x_op](x_head)
                    x_tail = self.t_ops_dict[x_op](x_tail)
                else:
                    x_head = x_tail = self.h_ops_dict[x_op](x_head)

        if len(r_ops) > 0:
            for r_op in r_ops.split("."):
                r = self.r_ops_dict[r_op](r)

        return x_head, x_tail, r


class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=8):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc = nn.Sequential(nn.Conv2d(in_planes, in_planes//ratio, 1, bias=False),
                                nn.ReLU(),
                                nn.Conv2d(in_planes//ratio, in_planes, 1, bias=False))
        self.sigmoid = nn.Tanh()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = avg_out + max_out
        return self.sigmoid(out)


class QConvE(LTEModel):
    def __init__(self, num_ents, num_rels, params=None):
        super(self.__class__, self).__init__(num_ents, num_rels, params)

        self.bn = torch.nn.BatchNorm1d(self.p.embed_dim)
        self.hidden_drop = torch.nn.Dropout(self.p.hid_drop)
        self.hidden_drop2 = torch.nn.Dropout(self.p.conve_hid_drop)
        self.feature_drop = torch.nn.Dropout(self.p.feat_drop)

        self.QuaternionConv = QuaternionConv(4, out_channels=self.p.num_filt,
                                             kernel_size=(self.p.ker_sz, self.p.ker_sz),
                                             stride=1, padding=0, bias=self.p.bias)

        self.linear = nn.Linear(self.p.embed_dim * 2, 4 * self.p.embed_dim * 2, bias=True)

        flat_sz_h = int(2 * self.p.k_w) - self.p.ker_sz + 1
        flat_sz_w = self.p.k_h - self.p.ker_sz + 1
        self.flat_sz = flat_sz_h * flat_sz_w * self.p.num_filt
        self.fc = torch.nn.Linear(self.flat_sz, self.p.embed_dim)
        self.Tanh = nn.Tanh()
        self.conv = QuaternionConv(in_channels=self.p.num_filt, out_channels=self.p.num_filt, kernel_size=1, stride=1,
                                   padding=0)
        self.channel_att = ChannelAttention(self.p.num_filt)

    def concat(self, e1_embed, rel_embed):
        s_a, x_a, y_a, z_a = torch.chunk(e1_embed, 4, dim=1)
        s_b, x_b, y_b, z_b = torch.chunk(rel_embed, 4, dim=1)

        A = torch.cat([s_a, s_b], dim=1)
        B = torch.cat([x_a, x_b], dim=1)
        C = torch.cat([y_a, y_b], dim=1)
        D = torch.cat([z_a, z_b], dim=1)

        stack_inp = torch.cat([A, B, C, D], dim=1)
        stack_inp = self.linear(stack_inp)
        stack_inp = stack_inp.reshape((-1, 4, 2 * self.p.k_w, self.p.k_h))
        return stack_inp

    def forward(self, sub, rel):
        x = self.init_embed
        r = self.init_rel

        x_h, x_t, r = self.exop(x, r, self.x_ops, self.r_ops)
        sub_emb = torch.index_select(x_h, 0, sub)
        rel_emb = torch.index_select(r, 0, rel)
        all_ent = x_t
        stk_inp = self.concat(sub_emb, rel_emb)
        x = q_normalize(stk_inp)
        x = self.QuaternionConv(x)
        x = q_normalize(x)
        x = self.channel_att(x) * x
        x = q_normalize(x)
        x = F.relu(x)
        x = self.feature_drop(x)

        x = x.view(-1, self.flat_sz)
        x = self.fc(x)
        x = self.hidden_drop2(x)
        x = self.bn(x)
        x = F.relu(x)

        x = torch.mm(x, all_ent.transpose(1, 0))
        x += self.bias.expand_as(x)

        return x
