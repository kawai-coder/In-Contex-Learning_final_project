import torch
import copy as cp 

class DataSampler:
    def __init__(self, n_dims):
        self.n_dims = n_dims

    def sample_xs(self):
        raise NotImplementedError
    

class SinSampler(DataSampler):
    '''
    Input:
        n_dim: dims of data points 
    Ouput:
        Ax + sin(Wx + b), where A and W come from the same matrix space  
    '''
    def __init__(self, n_dims, scale_out=None, scale_in=None, bias=None):
        '''
        Input:
            scale_out: refering to 'A' matrix; shape: n_dims x out_dims
            scale_in : refering to 'W' matrix; shape: n_dims x out_dims
            bias     : refering to b   vector; shape: 1      x out_dims  
        '''
        super().__init__(n_dims)
        self.scale_out = scale_out
        self.scale_in  = scale_in
        self.bias      = bias
        if self.scale_in is not None:
            assert scale_in.shape[0]  == n_dims, "The dim0 of scale matrix should match n_dims"
        if self.scale_out is not None:
            assert scale_out.shape[0] == n_dims, "The dim0 of scale matrix should match n_dims"
        if self.bias is not None:
            assert self.bias.shape[1] == scale_in.shape[1], "The dims1 of bias should match dim0 of scale_in"
        

    def sample_xs(self, n_points, b_size, n_dims_truncated=None, seeds=None):
        if seeds is None:
            xs_b = torch.randn(b_size, n_points, self.n_dims)
            xs_b_copy = cp.deepcopy(xs_b)
        else:
            xs_b = torch.zeros(b_size, n_points, self.n_dims)
            generator = torch.Generator()
            assert len(seeds) == b_size, "The len of seeds should be equal to batch_size"

            for i, seed in enumerate(seeds):
                generator.manual_seed(seed)
                xs_b[i] = torch.randn(n_points, self.n_dims, generator=generator) 
            
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

    test_y = SinSampler(
                    n_dims = 32, 
                    scale_out = torch.randn(32, 32),
                    scale_in  = torch.randn(32, 32),
                    bias      = torch.randn(1 , 32))

    seeds = [i**2 for i in range(64)]
    
    dataset_x = SinSampler(n_dims=32).sample_xs(32, 64, seeds=seeds)
    dataset_y = test_y.sample_xs(32, 64, seeds=seeds)

    print(dataset_x.shape, dataset_y.shape)
