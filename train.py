import torch
import torch.nn as nn
import torch.nn.functional as F 
import numpy as np 
import sampler 
import matplotlib.pyplot as plt
import wandb 
import model as ml 
from torch.utils.data import Dataset, DataLoader
import os
from datetime import datetime

class RandomSeededDataset(Dataset):
    def __init__(self, 
                 num_samples, 
                 num_points_per_sample, 
                 ys_DataSampler:sampler.DataSampler, 
                 xs_DataSampler:sampler.DataSampler, 
                 seeds=None, 
                 seed_max_value=9223372036854775807, 
                 device='cpu',
                 n_rank_range = np.linspace(2, 21, 20, endpoint=True)):
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
        self.n_rank_range = n_rank_range

        # Generate all seeds once
        if seeds is None:
            self.seeds = torch.empty(num_samples, dtype=torch.int64, device=device).random_(0, seed_max_value).tolist()
        else:
            ## Using the same seeds to generate xs
            ## We hope every 10k examples we change the rank of the data we generated  
            self.seeds = seeds
        
    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        seed = self.seeds[idx]
        # Generate random points and its corresbonding labels
        n_rank = (idx // 128000) % len(self.n_rank_range)
        ## here we increase the rank of the training data pts per 1280000 data points, approximately 1280000 / 64 = 2000 interaction to keep the same with the paper 

        points = self.xs_DataSampler.sample_xs(
            n_points = self.num_points,
            b_size   = 1,
            n_rank   = n_rank, ## for one thousand run we add the rank of the data pts  
            seeds    = [seed]
        )
        labels = self.ys_DataSampler.sample_xs(
            n_points = self.num_points,
            b_size   = 1,
            n_rank   = n_rank, 
            seeds    = [seed]
        ) 
        ## --> shape is 1 x self.num_points x n_dim
        return points[0], labels[0] ## avoid returning a 3d tensor which may raise error!

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
                 save_model_check_pt = False,
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
        self.save_model_check_pt = save_model_check_pt
        self.record_wandb = record_wandb
        self.visualize    = visualize
        self.checkpoint_path = check_point_path
        self.device       = next(self.model.parameters()).device


    def load_model(self):
        if self.checkpoint_path is not None:
            print(f"---Loading the pretrained params from {self.checkpoint_path}---")
            self.model.checkpoints_loading(self.model, self.checkpoint_path)

    def generate_random_seeds(batch_size, max_value=MAX_INT64, device='cpu', dtype=torch.int64):
        
        random_seeds = torch.randint(
            0, 
            max_value,
            size=(batch_size, ),
            device=device,
            dtype=dtype)
        
        return random_seeds.tolist() 
    
    @staticmethod
    def generate_checkpoint_name(epoch, loss, learning_rate, num_example, directory='checkpoints'):
    # Create the directory if it doesn't exist
        os.makedirs(directory, exist_ok=True)
        
        # Format the current date and time for uniqueness (optional)
        date_str = datetime.now().strftime("%Y-%m-%d_%H%M%S")
        
        # Generate the filename with all relevant information
        filename = (f"numOfExample={num_example}_epoch={epoch}_lr={learning_rate:.6f}_loss={loss:.4f}_"
                    f"{date_str}.pt")
        
        # Full path to save
        save_path = os.path.join(directory, filename)
        
        return save_path
    
    def __call__(self, lr:int, batch_size:int):
        if self.checkpoint_path:
            self.load_model()

        if self.record_wandb:
            wandb.init(
            project="in-context-learning",     # Change to your paper/project name
            name=f"{self.model.name}_{self.epochs}_runs",      # This name will appear in the dashboard
            config={
                "model": self.model.name,
                "embedding_dim": self.model._read_in.out_features,
                "lr": lr,
                "batch_size": batch_size,
                "epochs": self.epochs,
                "sampler":self.dataSampler
            }
        ) 
        if self.visualize:
            loss_record = []
            
        ## define the optimizer and create loss_eval to record the loss 
        optimizer = self.optimizer(self.model.parameters(), lr=lr, betas=(0.9, 0.95))
        ## scheduler setup 
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=40000)  # Decays LR over 40,000 steps
        
        device    = self.device
        loss_eval = 0 
        ## keep track of the total inter that the model has run 
        total_inter = 0
        # Train loop 
        for epoch in range(self.epochs): ## For each epoch we can change the training dataset
            ## Define the data_sampler
            Train_data_ys_sampler = self.dataSampler(
            n_dims   = self.n_dims,
            scale_in = torch.randn((self.n_dims, self.n_dims), device=self.device),
            bias     = torch.randn((1, self.n_dims), device = self.device)) 

            Train_data_xs_sampler = self.dataSampler(
            n_dims   = self.n_dims)

            ## Generate the dataset we use to train 
            dataset_train = RandomSeededDataset(
            num_samples = self.num_example,
            num_points_per_sample = 128,
            ys_DataSampler = Train_data_ys_sampler, 
            xs_DataSampler = Train_data_xs_sampler)

            dataloader_train = DataLoader(dataset_train, self.batch_size) ## Because here I change the way of the data generating.
            
            for batch_idx, (xs, ys) in enumerate(dataloader_train): 
                total_inter += 1
                ## ys.shape is self.batch x num_examples x n_dims 
                pred = self.model(xs, ys)
                loss = self.loss_fn(pred, ys)
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                loss_eval += loss.item()
                ## Perform the update on the scheduler
                scheduler.step()
        
                if (total_inter-1) % 50 == 0:
                    if total_inter == 1: 
                        avg_loss = loss_eval
                    else:
                        avg_loss = loss_eval / 50
                    print(f"Epoch:{epoch}, Inter:{batch_idx}, Current loss:{avg_loss:0.4f}")
                    loss_record.append((epoch, batch_idx, avg_loss))
                    loss_eval = 0 
     
                if self.record_wandb:
                    wandb.log({"train_loss": avg_loss, 
                               "step"      : self.epochs, 
                               "epoch"     : epoch})
        if self.visualize :
            epoch_list, batch_idx_list, avg_loss_list = zip(*loss_record)   
            num_iter_per_epoch = self.num_example // self.batch_size
            x_pos = [x*num_iter_per_epoch + y for x,y in zip(epoch_list, batch_idx_list)] 
            
            plt.plot(x_pos, avg_loss_list)    
            plt.xlabel("Train Inter")
            plt.ylabel("Train Loss")
            plt.title(f"Model({self.model.name})Train Loss Curve")
            plt.grid(True)
            plt.show()

        if self.save_model_check_pt:

            checkpoint = {
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': loss.item()}
            
            torch.save(checkpoint, Train_from_scratch.generate_checkpoint_name(self.epochs, loss.item(), lr, self.num_example, directory="model_check_pt\\Mamba"))

        if self.record_wandb:
            ## save the model checkpoint on the weight and bias website 
            artifact = wandb.Artifact('Mamba-checkpoint', type='model')
            artifact.add_file(Train_from_scratch.generate_checkpoint_name(self.epochs, loss.item(), lr, self.num_example, directory="model_check_pt\\Mamba"))
            wandb.log_artifact(artifact)

if __name__ == "__main__":

    model_mamba = ml.Mamba(
        n_dims = 20 
    )
    model_mamba.to('cuda')
    train_mamba = Train_from_scratch(
        model_mamba,
        torch.optim.AdamW,
        torch.nn.MSELoss(),
        epochs = 5,
        dataSampler = sampler.SinSampler_with_lowrank,
        num_example = 128000 * 20,
        num_points_per_example = 128,
        data_dim = 20,
        save_model_check_pt = True,
        record_wandb = True)

    train_mamba(lr=2e-4, batch_size=64)
    ## We want each rank has 64 * 2000 data points to train 
    ## so we just need 1000 interaction and then I can finish it 
    ## 
