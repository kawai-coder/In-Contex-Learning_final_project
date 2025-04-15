
from sampler import SinSampler
import torch
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def visualize_sin_sampler(sampler, n_points=200, seed=0):
    """
    Visualizes 2D input -> 2D or 3D output from a SinSampler.
    Works best when input dimension = 2 and output dimension = 2 or 3.
    """
    # Sample a single batch
    samples = sampler.sample_xs(n_points=n_points, b_size=1, seeds=[seed])[0]  # shape: [n_points, out_dims]

    out_dims = samples.shape[-1]
    
    if out_dims == 2:
        plt.figure(figsize=(6, 6))
        plt.scatter(samples[:, 0], samples[:, 1], alpha=0.7, c='teal', edgecolor='k')
        plt.title("SinSampler Output (2D)")
        plt.xlabel("Output dim 1")
        plt.ylabel("Output dim 2")
        plt.grid(True)
        plt.axis("equal")
        plt.show()

    elif out_dims == 3:
        fig = plt.figure(figsize=(8, 6))
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(samples[:, 0], samples[:, 1], samples[:, 2], c='orange', edgecolor='k', alpha=0.7)
        ax.set_title("SinSampler Output (3D)")
        ax.set_xlabel("Dim 1")
        ax.set_ylabel("Dim 2")
        ax.set_zlabel("Dim 3")
        plt.show()
        
    else:
        print(f"Output dim = {out_dims}. Only 2D or 3D output is supported for visualization.")


if __name__ == "__main__":
    # Example with 2D input and 2D output
    n_dims = 3 
    out_dims = 3
    sampler = SinSampler(
        n_dims=n_dims,
        scale_in=torch.randn(n_dims, out_dims),
        scale_out=torch.randn(n_dims, out_dims),
        bias=torch.randn(1, out_dims)
    )

    visualize_sin_sampler(sampler, n_points=300, seed=42)

