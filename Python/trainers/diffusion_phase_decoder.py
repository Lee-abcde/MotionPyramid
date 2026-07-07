# import torch
# from denoising_diffusion_pytorch import Unet, GaussianDiffusion, Trainer
# from models.ddpm import diffusion
#
#
#
# trainer = Trainer(
#     diffusion,
#     'path/to/your/images',
#     train_batch_size = 32,
#     train_lr = 8e-5,
#     train_num_steps = 700000,         # total training steps
#     gradient_accumulate_every = 2,    # gradient accumulation steps
#     ema_decay = 0.995,                # exponential moving average decay
#     amp = True,                       # turn on mixed precision
#     calculate_fid = True              # whether to calculate fid during training
# )
