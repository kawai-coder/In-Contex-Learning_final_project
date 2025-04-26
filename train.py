import torch
import torch.nn as nn
import torch.nn.functional
import numpy as np 
from transformers import GPT2Model, GPT2Config, MambaModel, MambaConfig
import sampler 
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
import wandb 
import model as ml 
from torch.utils.data import Dataset, DataLoader
import torch
from torch.utils.data import Dataset

class RandomSeededDataset(Dataset):
    def __init__(self, num_samples, num_points_per_sample, ys_DataSampler:sampler.DataSampler, xs_DataSampler, seeds=None, seed_max_value=9223372036854775807, device='cpu'):
        """
        Args:
            num_samples (int): Total number of samples.
            num_points_per_sample (int): How many random points per sample.
            seed_max_value (int, optional): Max value for random seeds.
            device (str, optional): Device to generate seeds.
        """
        self.num_samples = num_samples
        self.num_points = num_points_per_sample
        self.device = device
        self.xs_DataSampler = xs_DataSampler
        self.ys_DataSampler = ys_DataSampler

        # Generate all seeds once
        if seeds is None:
            self.seeds = torch.empty(num_samples, dtype=torch.int64, device=device).random_(0, seed_max_value).tolist()
        else:
            ## Using the same seeds to generate xs
            self.seeds = seeds

        
    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        seed = self.seeds[idx]
        # Generate random points and its corresbonding labels

        points = self.xs_DataSampler.sample_xs(
            n_points = self.num_points,
            b_size   = 1,
            seeds    = [seed]
        )
        labels = self.ys_DataSampler.sample_xs(
            n_points = self.num_points,
            b_size   = 1,
            seeds    = [seed]
        ) 
        ## --> shape is 1 x self.num_points x n_dim
        return points, labels

class Train_from_scratch():

    MAX_INT64 = 9223372036854775807 ## torch.int64
    def __init__(self, 
                 model:ml.Model_myself, 
                 optimizer:torch.optim.Optimizer, 
                 loss_fn, 
                 epochs:int, 
                 dataSampler:sampler.DataSampler, 
                 num_example,
                 batch_size = 64, 
                 num_points_per_example = 128,
                 data_dim = 20, ## [20, 50, 100] 
                 check_point_path=None, 
                 record_wandb=False, 
                 visualize=True):

        self.model        = model 
        self.optimizer    = optimizer
        self.loss_fn      = loss_fn
        self.epochs       = epochs
        self.dataSampler  = dataSampler
        self.num_example  = num_example
        self.num_points_per_example = num_points_per_example 
        self.batch_size   = batch_size
        self.n_dims     = data_dim
        self.record_wandb = record_wandb
        self.visualize    = visualize
        self.checkpoint_path = check_point_path
        self.device       = next(self.model.parameters()).device


    def load_model(self):
        if self.checkpoint_path is not None:
            print(f"---Loading the pretrained params from {self.checkpoint_path}---")
            self.model.checkpoints_loading(self.checkpoint_path)

    def generate_random_seeds(batch_size, max_value=MAX_INT64, device='cpu', dtype=torch.int64):
        
        random_seeds = torch.randint(
            0, 
            max_value,
            size=(batch_size, ),
            device=device,
            dtype=dtype)
        
        return random_seeds.tolist() 
      
    def __call__(self, lr, batch_size):
        if self.checkpoint_path:
            self.load_model()

        if self.record_wandb:
            wandb.init(
            project="in-context-learning",     # Change to your paper/project name
            name=f"{self.model.name}_{self.epochs}_runs",      # This name will appear in the dashboard
            config={
                "model": self.model.name,
                "embedding_dim": self.model._read_in.out_features,
                "n_layers": self.model._backbone.config.n_layer,
                "n_heads": self.model._backbone.config.n_head,
                "lr": lr,
                "batch_size": batch_size,
                "epochs": self.epochs,
                "sampler":self.dataSampler
            }
        ) 
        if self.visualize:
            loss_record = []
            
        ## define the optimizer and create loss_eval to record the loss 
        optimizer = self.optimizer(self.model.parameters(), lr=lr)
        device    = self.device
        loss_eval = 0 

        ## Define the data_sampler
        Train_data_ys_sampler = self.dataSampler(
            n_dims   = self.n_dims,
            scale_in = torch.randn((self.n_dims, self.n_dims), device=self.device) 
        ) 
        Train_data_xs_sampler = self.dataSampler(
            n_dims   = self.n_dims
        )
        ## Generate the dataset we use to train 
        dataset_train = RandomSeededDataset(
            num_samples = self.num_example,
            num_points_per_sample = 128,
            ys_DataSampler = Train_data_ys_sampler, 
            xs_DataSampler = Train_data_xs_sampler    
        )
        dataloader_train = DataLoader(dataset_train, self.batch_size, shuffle = True)
        ## keep track of the total inter that the model has run 
        total_inter = 0
        # Train loop 
        for epoch in range(self.epochs):
            for batch_idx, (xs, ys) in enumerate(dataloader_train): 
                total_inter += 1
                ## ys.shape is self.batch x num_examples x n_dims 
                
                pred = self.model(xs, ys)
                loss = self.loss_fn(pred, ys)
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                loss_eval += loss.item()
        
                if total_inter % 50 == 0:
                    avg_loss = loss_eval / 50
                    print(f"Epoch:{epoch}, Inter:{batch_idx}, Current loss:{avg_loss:0.4f}")
                    loss_record.append((epoch, batch_idx, avg_loss))
                    loss_eval = 0 
     
                if self.record_wandb:
                    wandb.log({"train_loss": avg_loss, 
                               "step"      : self.epochs, 
                               "epoch"     : epoch})
        if self.visualize:
            epoch_list, batch_idx_list, avg_loss_list = zip(*loss_record)   
            num_iter_per_epoch = self.num_example // self.batch_size
            x_pos = [x*num_iter_per_epoch + y for x,y in zip(epoch_list, batch_idx_list)] 

            plt.plot(x_pos, avg_loss_list)
            plt.xlabel("Train Inter")
            plt.ylabel("Train Loss")
            plt.title(f"Model({self.model.name})Train Loss Curve")
            plt.grid(True)
            plt.show()
        
        # torch.save({
        #     "model_state_dict":self.model.state_dict(),
        #     'optimizer_state_dict':optimizer.state_dict(),
        #     'loss':loss.item()
        # }, f'transfomer_checkpoint_on_only_sin_test.pt')

        # if self.record_wandb:
        #     artifact = wandb.Artifact('transformer-checkpoint', type='model')
        #     artifact.add_file(f'transfomer_checkpoint_on_only_sin{40}k.pt')
        #     wandb.log_artifact(artifact)

if __name__ == "__main__":

    # def test_dataset():

    #     scale_in = torch.randn((20, 20), device='cuda')
    #     Train_data_ys_sampler = sampler.SinSampler(
    #             n_dims   = 20,
    #             scale_in = scale_in 
    #         ) 
    #     Train_data_xs_sampler = sampler.SinSampler(
    #             n_dims   = 20
    #         )
    #         ## Generate the dataset we use to train 
    #     dataset_train = RandomSeededDataset(
    #             num_samples = 100,
    #             num_points_per_sample = 128,
    #             ys_DataSampler = Train_data_ys_sampler, 
    #             xs_DataSampler = Train_data_xs_sampler    
    #         )
    #     dataloader_train = DataLoader(dataset_train, batch_size=32, shuffle=True)

    #     for xs, ys in dataloader_train:
    #         print(xs.shape)  # (32,)
    #         print(ys.shape)  # (32, 10, 2)
    #         print((ys - xs @ scale_in).abs().max().item())
    #         break


    # test_dataset()
    model = ml.TransformerModel(n_dims=20,
                                n_positions=256)
    model.to('cuda')
    train_test = Train_from_scratch(model, 
                                    torch.optim.AdamW,
                                    torch.nn.MSELoss(),
                                    epochs = 5,
                                    dataSampler = sampler.SinSampler,
                                    num_example = 10000,
                                    num_points_per_example = 128,
                                    data_dim = 20)                                    # check_point_path='transfomer_checkpoint_on_only_sin15k.pt')
    
    train_test(lr=1e-4, batch_size=64)
