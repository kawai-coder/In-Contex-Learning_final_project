import torch
import torch.nn as nn
import torch.nn.functional
import numpy as np 
from transformers import GPT2Model, GPT2Config, MambaModel, MambaConfig
import sampler 
import matplotlib.pyplot as plt
import copy as cp

device = "cuda" if torch.cuda.is_available() else "cpu"

class Model_myself(nn.Module):
    def __init__(self, n_dims):
        super().__init__()
        self.n_dims = n_dims
    
    @staticmethod 
    def checkpoints_loading(model:nn.Module, checkpoints_path=None):
        assert checkpoints_path != None, "There is no available checkpoints here"
        # Notice nn.Module just a container which doesn't directly have a attribute device
        checkpoint = torch.load(checkpoints_path, map_location=next(model.parameters()).device)
        model.load_state_dict(checkpoint["model_state_dict"])
    
    @staticmethod
    def count_parameters(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)

    @staticmethod
    def combine_xy(xs, ys):
        '''
        Input:
            sample xs; shape: b x n x   n_dim 
            sample ys; shape: b x n x out_dim
        
        Output:
            single seq combining x and y:
                      shape: b x (2*n) x max(n_dim, out_dim)
        '''
        B, N, n_dim = xs.shape
        out_dim     = ys.shape[2]
        
        if n_dim == out_dim:
            
            output = torch.concat((xs, ys), dim=2) ## b x n x (n_dim + out_dim) = b x n x (2 * n_dim) 
            output = output.view(B, 2*N, n_dim)   
            return output    
        
        final_emb_dim = max(n_dim, out_dim)
        if final_emb_dim == n_dim:
        ## out_dim < n_dim 
            ys_pad_zeros = torch.concat((ys, 
                                         torch.zeros(size=(B, N, n_dim - out_dim), device=device)), dim = 2)
            ## ys_pad_zeros.shape is B x N x n_dim 
            output = torch.concat((xs, ys_pad_zeros), dim=2)
            output = output.view(B, 2*N, n_dim)
            return output 
        
        if final_emb_dim == out_dim:
            xs_pad_zeros = torch.concat((xs, 
                                         torch.zeros(size=(B, N, out_dim - n_dim), device=device)), dim = 2)
            output = torch.concat((xs_pad_zeros, ys), dim=2)
            output = output.view(B, 2*N, out_dim)
            return output
    
class TransformerModel(Model_myself):
    def __init__(self, n_dims, n_positions, n_embd=128, n_layer=12, n_head=4):
        super(TransformerModel, self).__init__(n_dims)
        configuration = GPT2Config(
            n_positions=2 * n_positions,
            n_embd=n_embd,
            n_layer=n_layer,
            n_head=n_head,
            resid_pdrop=0.0,
            embd_pdrop=0.0,
            attn_pdrop=0.0,
            use_cache=False,
        )
        self.name = f"gpt2_embd={n_embd}_layer={n_layer}_head={n_head}"

        self.n_positions = n_positions
        self._read_in = nn.Linear(self.n_dims, n_embd)
        self._backbone = GPT2Model(configuration)
        self._read_out = nn.Linear(n_embd, self.n_dims)

    def forward(self, xs, ys, return_hidden_state=False):
        xy_query = self.combine_xy(xs, ys)                                  ## shape is b x n x max(n_dim, out_dim)
        xy_emb = self._read_in(xy_query)                                    ## shape is b x n x emb_dim(default:128)
        output = self._backbone(inputs_embeds = xy_emb).last_hidden_state   ## shape is b x n x emb_dim(default:128) skip the wte
        pred   = self._read_out(output) 

        if return_hidden_state:                                            ## if return_hidden_state == true, we want to visualize the function that GPT2 learn
            return pred[:, ::2, :], output[:, ::2, :]                      ## shape is b x n x max(n_dim, out_dim) returns only the pred of xs' hidden state
                                           
        return pred[:, ::2, :] ## predict only on ys
    
class Sinelayer(nn.Module):
    def __init__(self, in_features, out_features, bias=True, o_omega=30):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features, bias=bias)

    def forward(self, x):
        return torch.sin(self.linear(x))
        
class SinMLP(Model_myself):
    '''
    ## Here is a MLP model using sin() as its activation and its inductive bias is to model the periodic data
    Input:
        layers: a list which contains each layer's neuron number 
    Expected input: b x num_pts x  
    '''
    def __init__(self, n_dims, layers:list):
        super().__init__(n_dims) 
        
        self.pre_layer = [self.n_dims] + layers ## self.n_dims is kind of read in in GPT2 model 
        self.aft_layer = layers + [self.n_dims]
        self.proj_layer = nn.Linear(self.pre_layer[0], self.aft_layer[0])
        self.pred_layer = nn.Linear(self.pre_layer[-1], self.aft_layer[-1])
        self.backbond_ = torch.nn.ModuleList([Sinelayer(num_pre, num_aft) for num_pre, num_aft in zip(self.pre_layer[1:-1], self.aft_layer[1:-1])])
        self.name = f"SinMLP_n_dims={self.n_dims}_layers={len(self.backbond_)}"
    
    def forward(self, xs, ys, return_hidden_state=False): 
        '''
        Just fit the input format of transformer model, we don't need ys here  
        '''
        xs = self.proj_layer(xs)

        for layer in self.backbond_:
            xs = layer(xs)
            
        hidden_state = xs.clone() ## kind of like deepcopy
        
        xs = self.pred_layer(xs)

        if return_hidden_state:
            return xs, hidden_state
        else:
            return xs 


class Mamba(Model_myself):
    def __init__(self, 
                 n_dims, 
                 hidden_size=256, 
                 num_hidden_layers=6, 
                 expand=2, 
                 state_size=16):
        super().__init__(n_dims)
        ## d_model for model dimension or rather(embedding size) 
        ## n_layers: Number of Mamba blocks 
        ## d_state: State dimension for the selective SSM
        ## Notice here we don't need the Mamba self web layer we just input our own embedding into the model   
        configuration = MambaConfig(
            vocab_size = 2, ## dummy value, not used, just to reduce the total params of the model 
            hidden_size = hidden_size,  
            num_hidden_layers = num_hidden_layers,
            expand = expand,
            state_size = state_size
        )
        self._read_in = nn.Linear(n_dims, hidden_size)
        self._backbone = MambaModel(configuration)
        self._read_out = nn.Linear(hidden_size, n_dims)

        self.name = f"Mamba model_hidden_state={hidden_size}_layers={num_hidden_layers}_state_size={state_size}"

    def forward(self, xs, ys, return_hidden_state=False):
        ## here we want to implement the combine xs and ys as the input seq into our model
        xy_query = self.combine_xy(xs, ys) 
        # shape is b x (2*seq_len) x emb_size
        xy_emb = self._read_in(xy_query)                                    ## shape is b x n x emb_dim(default:128)
        output = self._backbone(inputs_embeds = xy_emb).last_hidden_state   ## shape is b x n x emb_dim(default:128) skip the wte
        pred   = self._read_out(output)
        if return_hidden_state is True:
            ## we can consider the alignment between input and last hidden state is one by one
            return pred[:, ::2, :], output[:, ::2, :] ## only return the pred on xs and the last hidden state on xs 
        else:
            return pred[:, ::2, :] 

if __name__ == "__main__":

    model_mamba = Mamba(n_dims=20)
    model_mamba.to(device)
    dummy_xs = torch.randn((64, 128, 20), device='cuda')
    dummy_ys = torch.randn((64, 128, 20), device='cuda')
    output = model_mamba(dummy_xs, dummy_ys)
    # with torch.no_grad():
        # print(output.shape)
    non_embed_params = sum(p.numel() 
                           for name, p in model_mamba.named_parameters()
                           if "embedding" not in name and "lm_head" not in name)
    print(f"Params excluding embeddings: {non_embed_params:,}")
    
