import torch
from torch import nn
from torch.nn import functional as F
from torch.autograd import Variable
from lib.pytorch_misc import enumerate_by_image
import numpy as np
# from lib.tucker import Tucker
# import lib.ten_ring as tr

class Fusion(nn.Module):

    def __init__(self,
            input_dims,
            ):
        super(Fusion, self).__init__()
        self.input_dims = input_dims
        self.linear0 = nn.Sequential(nn.Linear(input_dims, 256), nn.Linear(256, 256*15))
        self.linear1 = nn.Sequential(nn.Linear(input_dims, 256), nn.Linear(256, 256*15))
        self.linear_out = nn.Linear(256, input_dims)
        # self.n_params = sum(p.numel() for p in self.parameters() if p.requires_grad)

    def forward(self, x1, x2):
        x1 = self.linear0(x1)
        x2 = self.linear1(x2)
        o = x1 * x2
        o = o.view(-1, 15, 256)
        o = torch.sum(o, 1)
        o = self.linear_out(o)
        # o = tr(o)
        return o

class Get_Atten_map_mc(nn.Module):

    def __init__(self, input_dims):
        super(Get_Atten_map_mc, self).__init__()
        self.input_dims = input_dims
        self.fusion1 = Fusion(input_dims=512)
        self.fusion2 = Fusion(input_dims=512)
        self.w = nn.Linear(self.input_dims, 1)

    def forward(self, obj_feats, rel_inds, union_feats, n_nodes):

        prod = self.fusion1(obj_feats[rel_inds[:, 1]], obj_feats[rel_inds[:, 2]])
        pair_f1_o = self.fusion2(prod, union_feats)
        atten_f = self.w(pair_f1_o)
        atten_tensor = Variable(torch.zeros(n_nodes, n_nodes, 1)).cuda().float()
        head = rel_inds[:, 1:].min()
        atten_tensor[rel_inds[:, 1] - head, rel_inds[:, 2] - head] += atten_f
        atten_tensor = F.sigmoid(atten_tensor)
        atten_tensor = atten_tensor * (1- Variable(torch.eye(n_nodes).float()).unsqueeze(-1).cuda())
        return atten_tensor/torch.sum(atten_tensor,1)
        # return atten_tensor/torch.sum(atten_tensor,1) + atten_tensor/torch.sum(atten_tensor, dim=1, keepdim=True)

def mc_matmul(tensor3d, mat):
    out = []
    for i in range(tensor3d.size(-1)):
        out.append(torch.mm(tensor3d[:, :, i], mat))
    return torch.cat(out, -1)


class LayerNorm(nn.Module):

    def __init__(self,
                 normal_shape,
                 gamma=True,
                 beta=True,
                 epsilon=1e-5):
        """Layer normalization layer
        See: [Layer Normalization](https://arxiv.org/pdf/1607.06450.pdf)
        :param normal_shape: The shape of the input tensor or the last dimension of the input tensor.
        :param gamma: Add a scale parameter if it is True.
        :param beta: Add an offset parameter if it is True.
        :param epsilon: Epsilon for calculating variance.

        Thanks to CyberZHG's code in https://github.com/CyberZHG/torch-layer-normalization.git .
        """
        super(LayerNorm, self).__init__()
        if isinstance(normal_shape, int):
            normal_shape = (normal_shape,)
        else:
            normal_shape = (normal_shape[-1],)
        self.normal_shape = torch.Size(normal_shape)
        self.epsilon = epsilon
        if gamma:
            self.gamma = nn.Parameter(torch.Tensor(*normal_shape))
        else:
            self.register_parameter('gamma', None)
        if beta:
            self.beta = nn.Parameter(torch.Tensor(*normal_shape))
        else:
            self.register_parameter('beta', None)
        self.reset_parameters()

    def reset_parameters(self):
        if self.gamma is not None:
            self.gamma.data.fill_(1)
        if self.beta is not None:
            self.beta.data.zero_()

    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        var = ((x - mean) ** 2).mean(dim=-1, keepdim=True)
        std = (var + self.epsilon).sqrt()
        y = (x - mean) / std
        if self.gamma is not None:
            y *= self.gamma
        if self.beta is not None:
            y += self.beta
        return y

    def extra_repr(self):
        return 'normal_shape={}, gamma={}, beta={}, epsilon={}'.format(
            self.normal_shape, self.gamma is not None, self.beta is not None, self.epsilon,
        )

class Message_Passing4OBJ(nn.Module):

    def __init__(self, input_dims):
        super(Message_Passing4OBJ, self).__init__()
        self.input_dims = input_dims
        self.trans = nn.Sequential(nn.Linear(self.input_dims, input_dims//4), 
                                    LayerNorm(self.input_dims//4), nn.ReLU(),
                                    nn.Linear(self.input_dims//4, self.input_dims))

        self.get_atten_tensor = Get_Atten_map_mc(self.input_dims)

        self.conv = nn.Sequential(nn.Linear(self.input_dims, self.input_dims//2),
                                    nn.ReLU())
        # self.conv = nn.Linear(self.input_dims, self.input_dims // 4) # use rel in the end.
        
    def forward(self, obj_feats, phr_feats, im_inds, rel_inds):

        num_img = int(im_inds.max()) + 1
        obj_indices_sets = [torch.nonzero(im_inds==i).data.squeeze() for i in range(num_img)]
        obj2obj_feats_sets = []
        rel_indices_sets = [torch.nonzero(rel_inds[:, 0]==i).squeeze() for i in range(num_img)]

        for i, obj_indices in enumerate(obj_indices_sets):
            entities_num = obj_indices.size(0)
            cur_obj_feats = obj_feats[obj_indices]

            # cur_obj_feats = tr(cur_obj_feats)
            rel_indices = rel_indices_sets[i]
            atten_tensor = self.get_atten_tensor(obj_feats, rel_inds[rel_indices], phr_feats[rel_indices], entities_num)
            atten_tentor_t = torch.transpose(atten_tensor,1,0)
            # atten_tentor_t = atten_tentor_t/torch.sum(atten_tentor_t,1)
            atten_tensor = torch.cat((atten_tensor,atten_tentor_t),-1)
            # atten_tensor = F.softmax(atten_tensor, 1)
            context_feats = mc_matmul(atten_tensor, self.conv(cur_obj_feats))
            # context_feats = F.relu(mc_matmul(atten_tensor, self.conv(cur_obj_feats))) # use relu in the end.
            obj2obj_feats_sets.append(self.trans(context_feats))

        return F.relu(obj_feats + torch.cat(obj2obj_feats_sets, 0))