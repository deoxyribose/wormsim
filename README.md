# Simulator-based unsupervised detection and tracking of worms

The model is based on https://coix.readthedocs.io/en/latest/examples/bmnist.html

The simulator is taken from https://github.com/kirkegaardlab/deeptangle

Data generation is in gen_worm_data.ipynb

The model, inference program and training code are in wormsim_encoder_only.ipynb


The model samples 
1. Worm coordinates from the simulator (6 (x,y)-coordinates per worm)
2. Locations in the image
3. Generates 28 x 28 frames for each worm for all time steps (interpolating the coordinates from 1)
4. Places the frames at the locations sampled in 2.

The inference is amortized population Gibbs sampling (http://proceedings.mlr.press/v119/wu20h/wu20h.pdf), with
1. A kernel per time step that proposes the location of each worm - all kernels are parametrized by the same encoder_where NN, which learns to map from the 64 x 64 frames to the mean and variance of the latent locations
2. A kernel that proposes the simulator parameters - parametrized by a GRU, which learns to invert the simulator, i.e. map from the T x 28 x 28 worm frames to the simulator parameters that produce that worm

Model is still far from convergence:
![](https://github.com/deoxyribose/wormsim/blob/main/worms.gif)
On the left is the original video, with frames over the inferred locations.
On the right is reconstructed video.
