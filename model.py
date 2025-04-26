import torch
import torch.nn as nn
import torch.nn.functional
import numpy as np 
from transformers import GPT2Model, GPT2Config, MambaModel, MambaConfig
import sampler 
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
import wandb 

device = "cuda" if torch.cuda.is_available() else "cpu"

class Model_myself(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def checkpoints_loading(self, checkpoints_path=None):
        assert checkpoints_path != None, "There is no available checkpoints here"
        # Notice nn.Module just a container which doesn't directly have a attribute device
        checkpoint = torch.load(checkpoints_path, map_location=next(self.parameters()).device)
        self.load_state_dict(checkpoint["model_state_dict"])
        

    

class TransformerModel(Model_myself):
    def __init__(self, n_dims, n_positions, n_embd=128, n_layer=12, n_head=4):
        super(TransformerModel, self).__init__()
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
        self.n_dims = n_dims
        self._read_in = nn.Linear(n_dims, n_embd)
        self._backbone = GPT2Model(configuration)
        self._read_out = nn.Linear(n_embd, n_dims)


    @staticmethod
    def combine_xy(xs, ys):
        global device
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
        
    def forward(self, xs, ys, return_hidden_state=False):
        xy_query = self.combine_xy(xs, ys)                                  ## shape is b x n x max(n_dim, out_dim)
        xy_emb = self._read_in(xy_query)                                    ## shape is b x n x emb_dim(default:128)
        output = self._backbone(inputs_embeds = xy_emb).last_hidden_state   ## shape is b x n x emb_dim(default:128) skip the wte
        pred   = self._read_out(output) 

        if return_hidden_state:                                            ## if return_hidden_state == true, we want to visualize the function that GPT2 learn
            return pred[:, ::2, :], output[:, ::2, :]                      ## shape is b x n x max(n_dim, out_dim) returns only the pred of xs' hidden state
                                           
        return pred[:, ::2, :] ## predict only on ys
    


class Mamba(nn.Module):
    def __init__(self):
        pass

class KNNModel(KNeighborsClassifier):
    def __init__(self, n_neighbors = 5, *, weights = "uniform", algorithm = "auto", leaf_size = 30, p = 2, metric = "minkowski", metric_params = None, n_jobs = None):
        super().__init__(n_neighbors, weights=weights, algorithm=algorithm, leaf_size=leaf_size, p=p, metric=metric, metric_params=metric_params, n_jobs=n_jobs)
