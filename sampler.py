import torch
import copy as cp 

class DataSampler:
    def __init__(self, n_dims, device="cuda"):
        self.n_dims = n_dims
        self.device = device

    def sample_xs(self):
        raise NotImplementedError
    

class SinSampler(DataSampler):
    '''
    Input:
        n_dim: dims of data points 
    Ouput:
        Ax + sin(Wx + b), where A and W come from the same matrix space  
    '''
    def __init__(self, 
                 n_dims, 
                 device = 'cuda' if torch.cuda.is_available() else 'cpu', 
                 scale_out = None, 
                 scale_in = None, 
                 bias = None):
        '''
        Input:
            scale_out: refering to 'A' matrix; shape: n_dims x out_dims
            scale_in : refering to 'W' matrix; shape: n_dims x out_dims
            bias     : refering to b   vector; shape: 1      x out_dims  
        '''
        super().__init__(n_dims, device)
        self.scale_out = scale_out
        self.scale_in  = scale_in
        self.bias      = bias

        if self.scale_in is not None:
            assert scale_in.shape[0]  == n_dims, "The dim0 of scale matrix should match n_dims"
        if self.scale_out is not None:
            assert scale_out.shape[0] == n_dims, "The dim0 of scale matrix should match n_dims"
        if self.bias is not None:
            assert self.bias.shape[1] == scale_in.shape[1], "The dims1 of bias should match n_dims"
        

    def sample_xs(self, n_points, b_size, n_dims_truncated=None, seeds=None):
        if seeds is None:
            xs_b = torch.randn((b_size, n_points, self.n_dims), device=self.device)
            xs_b_copy = cp.deepcopy(xs_b)

        else:
            xs_b = torch.zeros((b_size, n_points, self.n_dims), device=self.device)
            generator = torch.Generator(device=self.device)
            assert len(seeds) == b_size, "The len of seeds should be equal to batch_size"

            for i, seed in enumerate(seeds):
                generator.manual_seed(seed)
                xs_b[i] = torch.randn((n_points, self.n_dims), generator=generator, device=self.device) 
            
            xs_b_copy = cp.deepcopy(xs_b)
            
        if self.scale_in is not None:
            xs_b = xs_b @ self.scale_in
        if self.bias is not None:
            xs_b = torch.sin(xs_b + self.bias)
        if self.scale_out is not None:
            xs_b += xs_b_copy @ self.scale_out ## shape is b x n_points x out_dims
        if n_dims_truncated is not None:
            xs_b = xs_b[:, :, n_dims_truncated:]
            
        return xs_b


class GaussianNoiseSampler(SinSampler):
    '''
    On the basis of SinSampler we want to add some noise at each data pts 
        params:
            noise_scale: normal distribution mean
            noise_var  : normal distribution std 
    '''
    
    def __init__(self, n_dims, noise_scale=0, noise_std=1, scale_out=None, scale_in=None, bias=None, device="cuda"):
        super().__init__(n_dims, scale_out, scale_in, bias, device)
        self.noise_scale = noise_scale
        self.noise_std = noise_std

    def sample_xs(self, n_points, b_size, n_dims_truncated=None, seeds=None):
        
        xs_b = super().sample_xs(n_points, b_size, n_dims_truncated, seeds)
        ## xs_b.shape is b x n_pts x n_dim 
        noise = self.noise_scale + self.noise_std * torch.randn(xs_b.shape, device=self.device) 
        xs_b += noise
        return xs_b
    

class UniformSampler(DataSampler):
    '''
    Input: n_dims: as the dims of the input data point 
    Output: uniformly randomly picked up points from [-2pi, 2pi] to be a warm up task for the model to trian 
.?
    '''
    def __init__(self, 
                 n_dims, 
                 scale_in = None,
                 scale_out = None,
                 bias = None,
                 device = 'cuda' if torch.cuda.is_available() else 'cpu'):
        
        super().__init__(n_dims, device)
        self.scale_out = scale_out
        self.scale_in  = scale_in
        self.bias      = bias

        
    def sample_xs(self,
                  n_points : int,
                  b_size : int,
                  seeds = None): ## 4pi x - 2pi 
                                 ## here we use scale_in as our flag to indicate return xs or ys
        if seeds is not None:
            generator = torch.Generator(device=self.device)
            for seed_idx, seed in enumerate(seeds):
                generator.manual_seed(seed)
                xs = torch.zeros(b_size, n_points, self.n_dims, device = self.device)
                xs_i = 4 * torch.pi * torch.rand(1, n_points, self.n_dims, device = self.device, generator=generator) - 2 * torch.pi
                xs[seed_idx] = xs_i

        if seeds is None:
            xs = 4 * torch.pi * torch.rand(b_size, n_points, self.n_dims, device = self.device, generator=generator) - 2 * torch.pi
            
        if self.scale_in is not None:
            return torch.sin(xs)
        else:
            return xs
        ## pts uniformly distributed in the range of [-2pi, 2pi) 

class LowRankSamplerWithoutScale(DataSampler):
    
    def __init__(self, 
                 n_dims,
                 scale_in = None, ## here the scale_in.. params just to make the API keep the same
                 scale_out = None,
                 bias = None, 
                 n_rank = 10, 
                 device="cuda"):
        
        super(self).__init__(n_dims, device, scale_in, scale_out, bias)
        self.n_rank = n_rank ## the rank of the matrix this sampler will generate !

    def sample_xs(self, n_points, b_size, n_dims_truncated=None, seeds=None):

        if seeds is None:
            xs_b_v = torch.randn((b_size, n_points, self.n_rank, self.n_dims), device=self.device)
            xs_b_u = torch.randn((b_size, n_points, self.n_rank, self.n_dims), device=self.device)
            xs_b = xs_b_u @ xs_b_v.transpose(-1, -2) ## the matrix is not symmetric 
            xs_b_copy = cp.deepcopy(xs_b)

        else:
            xs_b = torch.zeros((b_size, n_points, self.n_dims), device=self.device)
            generator = torch.Generator(device=self.device)
            assert len(seeds) == b_size, "The len of seeds should be equal to batch_size"

            for i, seed in enumerate(seeds):
                generator.manual_seed(seed)
                xs_b_v = torch.randn((n_points, self.n_rank, self.n_dims), generator=generator, device=self.device) 
                xs_b_u = torch.randn((n_points, self.n_rank, self.n_dims), generator=generator, device=self.device)
                xs_b[i] = xs_b_u @ xs_b_v.transpose(-1, -2)
        
        if self.scale_in: 
            ## just to keep the api the same across different data sampler 
            return torch.sin(xs_b)
        else:
            return xs_b.squeeze()

class SinSampler_with_lowrank(SinSampler):
    ## this sampler I want to implement the curriculum learning process 

    def __init__(self, n_dims, device='cuda' if torch.cuda.is_available() else 'cpu', scale_out=None, scale_in=None, bias=None):
        super().__init__(n_dims, device, scale_out, scale_in, bias)
        ## the rank is defined as the subspace dim of the datapts I gonna to pick up  
        
    def sample_xs(self, n_points, b_size, n_rank = 3, n_dims_truncated=None, seeds=None):
        ## reduce the complex of the data points 
        ## I should be able to change the rank when I use the method sample_xs
        
        self.n_rank = n_rank 
        assert self.n_rank <= self.n_dims
        
        if seeds is None:
            xs_b_u = torch.randn((b_size, n_points, self.n_rank), device=self.device)
            basis = torch.randn((self.n_rank, self.n_dims), device=self.device)
            basis = torch.nn.functional.normalize(basis, dim=1) ## normalize the basis 
            xs_b = xs_b_u @ basis ## the matrix is symmetric and semi definite 

        else:
        
            xs_b = torch.zeros((b_size, n_points, self.n_dims), device=self.device)
            generator = torch.Generator(device=self.device)
            assert len(seeds) == b_size, "The len of seeds should be equal to batch_size"
            for i, seed in enumerate(seeds):
                generator.manual_seed(seed)## I need to mamually set the seed and then begin to generate my dataset!
                basis = torch.randn((self.n_rank, self.n_dims), device=self.device)
                basis = torch.nn.functional.normalize(basis, dim=1) ## normalize the basis 
                # xs_b_v = torch.randn((n_points, self.n_rank, self.n_dims), generator=generator, device=self.device) 
                xs_b_u = torch.randn((n_points, self.n_rank), generator=generator, device=self.device)
                xs_b[i] = xs_b_u @ basis

        xs_b_copy = cp.deepcopy(xs_b)
                    
        if self.scale_in is not None:
            xs_b = xs_b @ self.scale_in
        if self.bias is not None:
            xs_b = torch.sin(xs_b + self.bias)
        if self.scale_out is not None:
            xs_b += xs_b_copy @ self.scale_out ## shape is b x n_points x out_dims
        if n_dims_truncated is not None:
            xs_b = xs_b[:, :, n_dims_truncated:]

        return xs_b


if __name__ == "__main__":
    sampler_test = SinSampler_with_lowrank(n_dims=20)
    xs = sampler_test.sample_xs(n_points=128, b_size=64)
    print(xs.shape)
        

        
        
         
        
        

        
        
        

        
