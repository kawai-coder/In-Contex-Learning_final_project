import torch
import torch.nn as nn
import torch.nn.functional
from transformers import GPT2Model, GPT2Config
import sampler 
import matplotlib.pyplot as plt

device = "cuda" if torch.cuda.is_available() else "cpu"


class TransformerModel(nn.Module):
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
        
    def forward(self, xs, ys):
        xy_query = self.combine_xy(xs, ys)                                  ## shape is b x n x max(n_dim, out_dim)
        xy_emb = self._read_in(xy_query)                                    ## shape is b x n x emb_dim(default:128)
        output = self._backbone(inputs_embeds = xy_emb).last_hidden_state   ## shape is b x n x emb_dim(default:128) skip the wte
        pred   = self._read_out(output)                                     ## shape is b x n x max(n_dim, out_dim)
        return pred[:, ::2, :]

class Mamba(nn.Module):
    def __init__(self):
        pass

if __name__ == "__main__":

    ## x.shape is 32 x 256 x 64
    ## y.shape is 32 x 256 x 64

    model = TransformerModel(64, 256)
    model.to(device)
    
    loss_fn = torch.nn.MSELoss()
    optimizer_cls = torch.optim.AdamW

    def train_test(epochs, model, loss_fn, optimizer_cls):

        optimizer = optimizer_cls(model.parameters(), lr=1e-4)
        loss_record = []
        for epoch in range(epochs):
            for i in range(1000):
                ## each epoch we pass 128 
                seeds = torch.randint(0, 1000, size=(64,)).tolist()
                generator = torch.Generator(device=device)
                generator.manual_seed(1)
                y_sampler = sampler.SinSampler(
                    n_dims    = 32,
                    scale_in  = torch.randn((32, 64), generator=generator, device=device),
                    scale_out = torch.randn((32, 64), generator=generator, device=device),
                    bias      = torch.randn((1 , 64), generator=generator, device=device),
                    device    = device
                )
                ys = y_sampler.sample_xs(128, 64, seeds=seeds)
                xs = sampler.SinSampler(n_dims = 32).sample_xs(128, 64, seeds=seeds)
                 
                pred = model(xs, ys)
                loss = loss_fn(pred, ys)
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                loss_record.append(loss.item())
                if i % 50 == 0:
                    print(f"Inter:{i}, Current loss:{loss.item():0.4f}")

        plt.plot(list(range(len(loss_record))), loss_record)
        plt.xlabel("Train_iter")
        plt.ylabel("Train_loss")
        plt.show()


    train_test(1, model, loss_fn, optimizer_cls)
