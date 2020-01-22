import torch
import torch.nn as nn
from torch.distributions import Normal
from functools import reduce
from operator import mul
from collections import Iterable
import torch.nn.functional as F
from copy import deepcopy

class LateralBlock(nn.Module):
    def __init__(self,col,depth,block,out_shape, in_shapes):
        super(LateralBlock,self).__init__()
        self.col = col
        self.depth = depth
        self.out_shape = out_shape
        self.block = block
        self.u = nn.ModuleList()


        if self.depth > 0:
            red_in_shapes = [reduce(mul,in_shape) if isinstance(in_shape,Iterable) else in_shape for in_shape in in_shapes]
            red_out_shape = reduce(mul,out_shape) if isinstance(out_shape,Iterable) else out_shape
            self.u.extend([nn.Linear(in_shape,red_out_shape) for in_shape in red_in_shapes])


    def forward(self,inputs,activated = True):
        if not isinstance(inputs, list):
            inputs = [inputs]

        cur_column_out = self.block(inputs[-1])
        out_shape = tuple(j for i in (-1, self.out_shape) for j in (i if isinstance(i, tuple) else (i,)))
        prev_columns_out = [mod(x.view(x.shape[0],-1)).view(out_shape) for mod, x in zip(self.u, inputs)]
        res= cur_column_out + sum(prev_columns_out)
        if activated:
            res = F.relu(res)
        return res

class QNetwork(nn.Module):

    def __init__(self, input_size, action_size, hidden_layers_size):
        super().__init__()

        self.hiddens = nn.ModuleList([nn.Linear(input_size, hidden_layers_size[0])])
        for i in range(1, len(hidden_layers_size)):
            self.hiddens.append(nn.Linear(hidden_layers_size[i-1], hidden_layers_size[i]))
        self.output = nn.Linear(hidden_layers_size[-1], action_size)

    def forward(self, x):
        for layer in self.hiddens:
            x = torch.relu(layer(x))
        return self.output(x)

    def save(self, file):
        torch.save(self.state_dict(), file)

    def load(self, file, device):
        self.load_state_dict(torch.load(file, map_location=device))


class CriticNetwork(nn.Module):
    def __init__(self, state_size, action_size, hidden_layers_size):
        super().__init__()

        self.hiddens = nn.ModuleList([nn.Linear(state_size+action_size, hidden_layers_size[0])])
        for i in range(1, len(hidden_layers_size)):
            self.hiddens.append(nn.Linear(hidden_layers_size[i-1], hidden_layers_size[i]))
        self.output = nn.Linear(hidden_layers_size[-1], 1)

    def forward(self, state, action):
        x = torch.cat([state, action], -1)
        for layer in self.hiddens:
            x = torch.relu(layer(x))
        return self.output(x)

    def save(self, file):
        torch.save(self.state_dict(), file)

    def load(self, file, device):
        self.load_state_dict(torch.load(file, map_location=device))


class CriticNetworkProgressive(CriticNetwork):
    def __init__(self, state_size, action_size, hidden_layers_size):
        super(CriticNetworkProgressive,self).__init__(state_size,action_size,hidden_layers_size)
        self.columns = nn.ModuleList([])
        self.depth = len(hidden_layers_size) + 1
        self.shapes = hidden_layers_size + [1]
        self.new_task(nn.Sequential(*self.hiddens,self.output),self.shapes)
        

    def forward(self,state,action,task_id = - 1):
        x = torch.cat([state, action], -1)
        return self.forward_prog(x,task_id = task_id)

    def forward_prog(self,x,task_id=-1):
        assert self.columns
        inputs = [col[0](x) for col in self.columns]
        for l in range(1,self.depth):
            out = []
            for i,col in enumerate(self.columns):
                out.append(col[l](inputs[:i+1],activated = (l== self.depth - 1)))

            inputs = out
        return out[task_id]

    def load(self,file,device,ntask = False):
        self.load_state_dict(torch.load(file, map_location=device))
        if ntask:
            self.freeze_columns()
            self.new_task()

    def new_task(self,new_layers=None,shapes=None):
        if shapes is None:
            shapes = self.shapes

        if new_layers is None:
            new_layers = deepcopy(nn.Sequential(*self.hiddens,self.output))
            for l in new_layers:
                l.reset_parameters()

        assert isinstance(new_layers,nn.Sequential)
        assert(len(new_layers) == len(shapes))

        task_id = len(self.columns)
        idx =[i for i,layer in enumerate(new_layers) if isinstance(layer,(nn.Linear))] + [len(new_layers)]
        new_blocks = []

        for k in range(len(idx) -1):
            prev_blocks = []
            if k > 0:
                prev_blocks = [col[k-1] for col in self.columns]

            new_blocks.append(LateralBlock(col = task_id,
                                           depth = k,
                                           block = new_layers[idx[k]:idx[k+1]],
                                           out_shape = shapes[idx[k+1]-1],
                                           in_shapes = self._get_out_shape_blocks(prev_blocks)
                                          ))

        new_column = nn.ModuleList(new_blocks)
        self.columns.append(new_column)



    def _get_out_shape_blocks(self,blocks):
        assert isinstance(blocks,list)
        assert all(isinstance(block,LateralBlock) for block in blocks)
        return [block.out_shape for block in blocks]



    def freeze_columns(self, skip=None):
        if skip == None:
            skip = []

        for i, c in enumerate(self.columns):
            if i not in skip:
                for params in c.parameters():
                    params.requires_grad = False

class ActorNetwork(nn.Module):
    def __init__(self, state_size, action_size, hidden_layers_size):
        super().__init__()
        self.hiddens = nn.ModuleList([nn.Linear(state_size, hidden_layers_size[0])])
        for i in range(1, len(hidden_layers_size)):
            self.hiddens.append(nn.Linear(hidden_layers_size[i-1], hidden_layers_size[i]))
        self.output = nn.Linear(hidden_layers_size[-1], action_size)

    def forward(self, x):
        for layer in self.hiddens:
            x = torch.relu(layer(x))
        return torch.tanh(self.output(x))

    def save(self, file):
        torch.save(self.state_dict(), file)

    def load(self, file, device):
        self.load_state_dict(torch.load(file, map_location=device))


class ValueNetwork(nn.Module):
    def __init__(self, state_size, hidden_layers_size):
        super().__init__()

        self.hiddens = nn.ModuleList([nn.Linear(state_size, hidden_layers_size[0])])
        for i in range(1, len(hidden_layers_size)):
            self.hiddens.append(nn.Linear(hidden_layers_size[i-1], hidden_layers_size[i]))
        self.output = nn.Linear(hidden_layers_size[-1], 1)

    def forward(self, x):
        for layer in self.hiddens:
            x = torch.relu(layer(x))
        return self.output(x)

    def save(self, file):
        torch.save(self.state_dict(), file)

    def load(self, file, device):
        self.load_state_dict(torch.load(file, map_location=device))


class SoftActorNetwork(nn.Module):
    def __init__(self, state_size, action_size, hidden_layers_size, device,
                 init_w=3e-3, log_std_min=-20, log_std_max=2):
        super().__init__()
        self.device = device
        self.action_size = action_size

        self.log_std_min = log_std_min
        self.log_std_max = log_std_max

        self.hiddens = nn.ModuleList([nn.Linear(state_size, hidden_layers_size[0])])
        for i in range(1, len(hidden_layers_size)):
            self.hiddens.append(nn.Linear(hidden_layers_size[i-1], hidden_layers_size[i]))

        self.mean_output = nn.Linear(hidden_layers_size[-1], action_size)
        self.mean_output.weight.data.uniform_(-init_w, init_w)
        self.mean_output.bias.data.uniform_(-init_w, init_w)

        self.log_std_output = nn.Linear(hidden_layers_size[-1], action_size)
        self.log_std_output.weight.data.uniform_(-init_w, init_w)
        self.log_std_output.bias.data.uniform_(-init_w, init_w)

        self.normal = Normal(0, 1)

    def forward(self, x):
        for layer in self.hiddens:
            x = torch.relu(layer(x))

        mean = self.mean_output(x)
        log_std = torch.tanh(self.log_std_output(x))
        log_std = self.log_std_min + (self.log_std_max - self.log_std_min) * (log_std+1) / 2

        return mean, log_std

    def evaluate(self, state):
        mean, log_std = self(state)
        std = log_std.exp()

        z = self.normal.sample((self.action_size, )).to(self.device)
        action = torch.tanh(mean + std*z)

        log_prob = Normal(mean, std).log_prob(mean + std*z).sum(dim=1).unsqueeze(1)
        # Cf Annexe C.
        log_prob -= torch.log(1 - action.pow(2) + 1e-6).sum(dim=1).unsqueeze(1)

        return action, log_prob

    def get_mu_sig(self, state):
        with torch.no_grad():
            mean, log_std = self(state)
        std = log_std.exp()
        return mean.cpu().numpy(), std.cpu().numpy()

    def select_action(self, state):
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            mean, log_std = self(state)
            std = log_std.exp()

            z = self.normal.sample((self.action_size, )).to(self.device)
            action = torch.tanh(mean + std*z)

        action = action.cpu().numpy()
        return action[0]

    def save(self, file):
        torch.save(self.state_dict(), file)

    def load(self, file, device):
        self.load_state_dict(torch.load(file, map_location=device))


class SoftActorNetworkProgressive(SoftActorNetwork):
    def __init__(self, state_size, action_size, hidden_layers_size, device,
                 init_w=3e-3, log_std_min=-20, log_std_max=2):
        super(SoftActorNetworkProgressive,self).__init__(state_size,action_size,hidden_layers_size,device,init_w,log_std_min,log_std_max)
        self.columns = nn.ModuleList([])
        self.depth = len(hidden_layers_size)
        self.shapes = hidden_layers_size
        self.mean_output = nn.ModuleList([self.mean_output])
        self.log_std_output = nn.ModuleList([self.log_std_output])
        self.state_size = state_size
        self.new_task(nn.Sequential(*self.hiddens),self.shapes)

    def forward(self, x, task_id = -1):
        x = self.forward_prog(x,task_id = task_id)
        mean = self.mean_output[task_id](x)
        log_std = torch.tanh(self.log_std_output[task_id](x))
        log_std = self.log_std_min + (self.log_std_max - self.log_std_min) * (log_std+1) / 2
        return mean, log_std

    def forward_prog(self,x,task_id=-1):
        assert self.columns
        inputs = [col[0](x) for col in self.columns]
        for l in range(1,self.depth):
            out = []
            for i,col in enumerate(self.columns):
                out.append(col[l](inputs[:i+1],activated = (l== self.depth - 1)))
            inputs = out
        return out[task_id]

    def load(self,file,device,ntask = False):
        self.load_state_dict(torch.load(file, map_location=device))
        if ntask:
            self.freeze_columns()
            self.new_task()

    def new_task(self,new_layers=None,shapes=None):
        if shapes is None:
            shapes = self.shapes

        if new_layers is None:
            new_network = SoftActorNetwork(self.state_size,self.action_size,self.shapes,self.device)
            self.mean_output.append(new_network.mean_output)
            self.log_std_output.append(new_network.log_std_output)
            new_layers = nn.Sequential(*new_network.hiddens)

        assert isinstance(new_layers,nn.Sequential)
        assert(len(new_layers) == len(shapes))

        task_id = len(self.columns)
        idx =[i for i,layer in enumerate(new_layers) if isinstance(layer,(nn.Linear))] + [len(new_layers)]
        new_blocks = []

        for k in range(len(idx) -1):
            prev_blocks = []
            if k > 0:
                prev_blocks = [col[k-1] for col in self.columns]

            new_blocks.append(LateralBlock(col = task_id,
                                           depth = k,
                                           block = new_layers[idx[k]:idx[k+1]],
                                           out_shape = shapes[idx[k+1]-1],
                                           in_shapes = self._get_out_shape_blocks(prev_blocks)
                                          ))

        new_column = nn.ModuleList(new_blocks)
        self.columns.append(new_column)



    def _get_out_shape_blocks(self,blocks):
        assert isinstance(blocks,list)
        assert all(isinstance(block,LateralBlock) for block in blocks)
        return [block.out_shape for block in blocks]



    def freeze_columns(self, skip=None):
        if skip == None:
            skip = []

        for i, c in enumerate(self.columns):
            if i not in skip:
                for params in c.parameters():
                    params.requires_grad = False

        for i,(mean,log_std) in enumerate(zip(self.mean_output,self.log_std_output)):
            if i not in skip:
                for params in mean.parameters():
                    params.requires_grad = False
                for params in log_std.parameters():
                    params.requires_grad = False
