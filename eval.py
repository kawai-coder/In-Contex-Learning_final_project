from model import *
from sampler import * 
from train import * 
from sklearn.neighbors import KNeighborsRegressor
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
import torch 
import numpy as np 
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import matplotlib.cm as cm 
from sklearn.linear_model import LinearRegression
            
class Compare_model():

    def __init__(self, dataSampler:sampler.DataSampler, number_example_range, visualize_option=["PCA", "t-SNE", "fft"]):
        '''
        Init:
            1.DataSampler: see in sampler.py 
            2.
        '''
        self.datasampler = dataSampler
        assert len(number_example_range) == 2, "You should enter a example with len = 2"
        self.number_range = np.linspace(number_example_range[0], number_example_range[1], 
                                        num = number_example_range[1] - number_example_range[0] + 1,
                                        endpoint=True, dtype=np.int32)
        
        self.visualize_option = visualize_option
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    def compare_drawing_eval_loss_pic(self, 
                            model_list:list, 
                            n_dims = 20, 
                            batch_size = 128
                            ):
        '''
        Input: 
            model_list[0]
            model_list[1]
        Output:
            visualize the plot
            Notice that the paper use the in_context_learning examples number as the x axis  
        '''
        device = next(model_list[0].parameters()).device
        
        seed_max_value = 9223372036854775807 ## torch.int64 maxmimun        
        baseline_model = KNeighborsRegressor(n_neighbors=3)
        eval_error_record = {f"{_}":[] for _ in range(len(model_list)+1)} # we use a dict to record the eval square error  
        loss_fn = torch.nn.MSELoss()
        
        for num in self.number_range:

            seeds = torch.empty(batch_size, dtype=torch.int64).random_(0, seed_max_value).tolist()
            
            ys_full = self.datasampler(n_dims = n_dims,
                            scale_in = torch.randn((n_dims, n_dims), device=device),
                            bias     = torch.randn((1, n_dims), device = device)).sample_xs(n_points=2*num, b_size=batch_size, seeds=seeds)
            
            xs_full = self.datasampler(n_dims = n_dims).sample_xs(n_points=2*num, b_size = batch_size, seeds=seeds)  
            
            xs_context = xs_full[:, :num, :]
            ys_context = ys_full[:, :num, :]
            
            xs_query   = xs_full[:, num:, :]
            ys_query   = torch.zeros(ys_full[:, num:, :].shape, device=device)
            true_y     = ys_full[:, num:, :]
            
            xs_input   = torch.concat((xs_context, xs_query), dim=1)
            ys_input   = torch.concat((ys_context, ys_query), dim=1) 
        
            print(f"eval is now in {num} steps")

            xs_context_2d = xs_context.reshape(-1, xs_context.shape[-1]).cpu().numpy()
            ys_context_2d = ys_context.reshape(-1, ys_context.shape[-1]).cpu().numpy()
            xs_query_2d   = xs_query.reshape(-1, xs_query.shape[-1]).cpu().numpy()
            true_y_2d = ys_full[:, num:, :].reshape(-1, ys_full[:, num:, :].shape[-1]).cpu().numpy()

            baseline_model.fit(xs_context_2d, ys_context_2d)
            pred_baselineModel = baseline_model.predict(xs_query_2d) ## number of examples x n_dims 
        
            baseline_model_error = mean_squared_error(true_y_2d, pred_baselineModel)
            eval_error_record['0'].append(baseline_model_error/n_dims)  ## here we normalize with the dim of the data pts
            
            for model_id, model in enumerate(model_list):

                with torch.no_grad():
                    pred_model = model(xs_input, ys_input)  ## here I set the first model is SinMLP; output has the same shape as ys_query 
                ## but we just need the the latter part of the output  
                    eval_error = loss_fn(pred_model[:, num:, :], true_y)
                    eval_error_record[f'{model_id+1}'].append(eval_error.item()/n_dims)
                ## Record the eval error of each model              
            # Clean up
            del xs_full, ys_full, xs_context, ys_context, xs_query, ys_query, true_y_2d, xs_context_2d, ys_context_2d
            del xs_input, ys_input, eval_error, xs_query_2d, baseline_model_error
            torch.cuda.empty_cache() ### to prevent the cuda out of memory!

        model_num = len(model_list) + 1 ## included the baseline model 
        plt.style.use('seaborn-v0_8')
        plt.figure(figsize=(6, 4), dpi=200)  # width x height in inches
        colors = cm.viridis(np.linspace(0, 1, model_num))  # Generates a range of colors
        
        for model_id, model in enumerate([baseline_model] + model_list):
            if model_id == 0:
                plt.plot(self.number_range, eval_error_record[f'{model_id}'], color = colors[model_id], label="baseline model(KNN=3)", linewidth=1.5)
            else:
                plt.plot(self.number_range, eval_error_record[f'{model_id}'], color = colors[model_id], label=f"{model.name}", linewidth=1.5)
        
        plt.xlabel("Number of in context examples")
        plt.ylabel("Square Loss")
        plt.legend()
        plt.title("Evaluating the trained Transformer on in-context-learning periodic function")
        plt.tight_layout()
        plt.grid(True)         
        plt.show()
        ## here we use one half of the data pts to be the in context example and use another half to be the query 

    def draw_hidden_state(self, 
                          model:nn.Module,
                          n_dims = 20, 
                          number_example = None,
                          dataSampler = sampler.SinSampler,
                          color_by = "mean"):
        
        seed_max_value = 9223372036854775807 ## torch.int64 maxmimun 
        if number_example is None:
            number_example = self.number_range[-1]

        seeds = torch.randint(1, seed_max_value, (number_example, )).tolist() 
        xs = dataSampler(n_dims=n_dims).sample_xs(
                                                n_points = 128, 
                                                b_size = number_example,
                                                n_rank = 20,
                                                seeds=seeds)   
        ys = dataSampler(n_dims   = n_dims,
                         scale_in = torch.randn((n_dims, n_dims), device = self.device),
                         bias     = torch.randn((1, n_dims), device = self.device)).sample_xs(n_points = 128, n_rank = 20, b_size=number_example, seeds=seeds)
        
        with torch.no_grad():
            pred, hidden_state = model(xs, ys, return_hidden_state = True)
        
        hidden_state_numpy = hidden_state.reshape(-1, hidden_state.shape[-1]).cpu().numpy() ## num_of_example x n_dims 
        hidden_state_pca_2d = PCA(n_components = 2).fit_transform(hidden_state_numpy)
        
        ys_flat = ys.reshape(-1, ys.shape[-1]).cpu().numpy()
        pred_flat = pred.reshape(-1, pred.shape[-1]).cpu().numpy() 

        if color_by == 'true_y':
            color_values = ys_flat.mean(axis=1) 
            color_label = "true_y (mean)"

        elif color_by == 'mean':
            color_values = np.linalg.norm(ys_flat - pred_flat, axis=1)
            color_label = "Prediction Error (L2)"

        plt.style.use('seaborn-v0_8')
        plt.figure(figsize=(6, 4), dpi=200)

       # Apply t-SNE
        tsne = TSNE(n_components=2, random_state=42, perplexity=30)
        hidden_state_tsne_2d = tsne.fit_transform(hidden_state_numpy)

        # Plot side-by-side: PCA (left), t-SNE (right)
        plt.style.use('seaborn-v0_8')
        fig, axes = plt.subplots(1, 2, figsize=(12, 4), dpi=200)

        # PCA Plot
        sc1 = axes[0].scatter(
            hidden_state_pca_2d[:, 0], hidden_state_pca_2d[:, 1],
            c=color_values, cmap='viridis', s=10
        )
        axes[0].set_title(f"PCA of Last Hidden State(colorby{color_label})")
        axes[0].set_xlabel("PC 1")
        axes[0].set_ylabel("PC 2")
        axes[0].grid(True)
        plt.colorbar(sc1, ax=axes[0], label='Color Scale')

        # t-SNE Plot
        sc2 = axes[1].scatter(
            hidden_state_tsne_2d[:, 0], hidden_state_tsne_2d[:, 1],
            c=color_values, cmap='viridis', s=10
        )
        axes[1].set_title(f"t-SNE of Last Hidden State(colorby{color_label})")
        axes[1].set_xlabel("Dim 1")
        axes[1].set_ylabel("Dim 2")
        axes[1].grid(True)
        plt.colorbar(sc2, ax=axes[1], label='Color Scale')

        plt.tight_layout()
        plt.show()
        
        
if __name__ == "__main__":
    
    model_SinMLP = SinMLP(20, [1024]*9)
    model_SinMLP.to('cuda')
    checkpoint_sinmlp = torch.load("model_check_pt/linear/numOfExample=100000_epoch=5_lr=0.000100_loss=0.4267_2025-04-28_022817.pt", map_location='cuda')
    model_SinMLP.load_state_dict(checkpoint_sinmlp['model_state_dict'])

    model_transformer = TransformerModel(20, 
                                         n_positions=256)
    model_transformer.to("cuda")
    checkpoint_transformer = torch.load("model_check_pt/gpt2/numOfExample=100000_epoch=5_lr=0.000100_loss=0.3548_2025-04-27_143010.pt", map_location='cuda')
    model_transformer.load_state_dict(checkpoint_transformer['model_state_dict'])
    model_Mamba = Mamba(
        n_dims = 20)
    model_Mamba.to('cuda')
    checkpoint_mamba = torch.load("model_check_pt\Mamba/numOfExample=10000_epoch=5_lr=0.000100_loss=0.4326_2025-05-03_013422.pt", map_location = "cuda")
    model_Mamba.load_state_dict(checkpoint_mamba['model_state_dict'])

    compare = Compare_model(dataSampler = sampler.SinSampler_with_lowrank,
                            number_example_range=[1, 50])
    compare.compare_drawing_eval_loss_pic([model_SinMLP,
                                model_Mamba,
                                model_transformer])
    
    # compare.draw_hidden_state(model_Mamba)
    ## Maybe training is not enough to make the model converge to a specific pt


