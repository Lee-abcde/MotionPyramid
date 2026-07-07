import torch
from argparse import ArgumentParser
from models.mdm import MDM
from diffusion import gaussian_diffusion as gd
from diffusion.respace import SpacedDiffusion, space_timesteps
from dataset import Text2PhaseData

def get_cond_mode(args):
    if args.unconstrained:
        cond_mode = 'no_cond'
    elif args.dataset in ['kit', 'humanml', 'text2phase']:
        cond_mode = 'text'
    else:
        cond_mode = 'action'
    return cond_mode
def add_base_options(parser):
    group = parser.add_argument_group('base')
    group.add_argument("--cuda", default=True, type=bool, help="Use cuda device, otherwise use CPU.")
    group.add_argument("--device", default=0, type=int, help="Device id to use.")
    group.add_argument("--seed", default=10, type=int, help="For fixing random seed.")
    group.add_argument("--batch_size", default=64, type=int, help="Batch size during training.")


def add_diffusion_options(parser):
    group = parser.add_argument_group('diffusion')
    group.add_argument("--noise_schedule", default='cosine', choices=['linear', 'cosine'], type=str,
                       help="Noise schedule type")
    group.add_argument("--diffusion_steps", default=1000, type=int,
                       help="Number of diffusion steps (denoted T in the paper)")
    group.add_argument("--sigma_small", default=True, type=bool, help="Use smaller sigma values.")


def add_model_options(parser):
    group = parser.add_argument_group('model')
    group.add_argument("--arch", default='trans_enc',
                       choices=['trans_enc', 'trans_dec', 'gru', 'unet', 'mlp'], type=str,
                       help="Architecture types as reported in the paper.")
    group.add_argument("--emb_trans_dec", default=False, type=bool,
                       help="For trans_dec architecture only, if true, will inject condition as a class token"
                            " (in addition to cross-attention).")
    group.add_argument("--layers", default=8, type=int,
                       help="Number of layers.")
    group.add_argument("--latent_dim", default=512, type=int,
                       help="Transformer/GRU width.")
    group.add_argument("--cond_mask_prob", default=.1, type=float,
                       help="The probability of masking the condition during training."
                            " For classifier-free guidance learning.")
    group.add_argument("--lambda_rcxyz", default=0.0, type=float, help="Joint positions loss.")
    group.add_argument("--lambda_vel", default=0.0, type=float, help="Joint velocity loss.")
    group.add_argument("--lambda_fc", default=1.0, type=float, help="Foot contact loss weight. Set to 0 to disable sliding loss.")
    group.add_argument("--unconstrained", action='store_true', default=False,
                       help="Model is trained unconditionally. That is, it is constrained by neither text nor action. "
                            "Currently tested on HumanAct12 only.")
    group.add_argument("--load_pretrained_diffusion_model", default=False, type=bool,
                       help="For trans_dec architecture only, if true, will inject condition as a class token"
                            " (in addition to cross-attention).")
    group.add_argument("--use_manifold", action='store_true', default=False,
                       help="Use continuous manifold embeddings instead of one-hot vectors for text2phase diffusion.")


def add_data_options(parser):
    group = parser.add_argument_group('dataset')
    group.add_argument("--dataset", default='humanml', choices=['humanml', 'kit', 'humanact12', 'uestc', 'text2phase'], type=str,
                       help="Dataset name (choose from list).")
    group.add_argument("--data_dir", default="", type=str,
                       help="If empty, will use defaults according to the specified dataset.")
    group.add_argument('--load', type=str, default="./Datasets/ljy1withRoot/Dataset-human-loco-gen2")
    group.add_argument('--std_cap', type=float, default=1e-7)
    group.add_argument('--window', type=float, default=4.0)
    # group.add_argument('--save', type=str, default="./results/test")


def add_training_options(parser):
    group = parser.add_argument_group('training')
    # group.add_argument("--diff_save_dir", default='./results/difftest1', type=str,
    #                    help="Path to save checkpoints and results.")
    group.add_argument("--diff_save_dir", required = True, type=str,
                       help="Path to save checkpoints and results.")
    group.add_argument("--overwrite", action='store_true',
                       help="If True, will enable to use an already existing save_dir.")
    group.add_argument("--train_platform_type", default='NoPlatform', choices=['NoPlatform', 'ClearmlPlatform', 'TensorboardPlatform'], type=str,
                       help="Choose platform to log results. NoPlatform means no logging.")
    group.add_argument("--lr", default=1e-4, type=float, help="Learning rate.")
    group.add_argument("--weight_decay", default=0.0, type=float, help="Optimizer weight decay.")
    group.add_argument("--lr_anneal_steps", default=0, type=int, help="Number of learning rate anneal steps.")
    group.add_argument("--eval_batch_size", default=32, type=int,
                       help="Batch size during evaluation loop. Do not change this unless you know what you are doing. "
                            "T2m precision calculation is based on fixed batch size 32.")
    group.add_argument("--eval_split", default='test', choices=['val', 'test'], type=str,
                       help="Which split to evaluate on during training.")
    group.add_argument("--eval_during_training", action='store_true',
                       help="If True, will run evaluation during training.")
    group.add_argument("--eval_rep_times", default=3, type=int,
                       help="Number of repetitions for evaluation loop during training.")
    group.add_argument("--eval_num_samples", default=1_000, type=int,
                       help="If -1, will use all samples in the specified split.")
    group.add_argument("--log_interval", default=1_000, type=int,
                       help="Log losses each N steps")
    group.add_argument("--save_interval", default=50_000, type=int,
                       help="Save checkpoints and run evaluation each N steps")
    group.add_argument("--num_steps", default=600_000, type=int,
                       help="Training will stop after the specified number of steps.")
    group.add_argument("--num_frames", default=60, type=int,
                       help="Limit for the maximal number of frames. In HumanML3D and KIT this field is ignored.")
    group.add_argument("--resume_checkpoint", default="", type=str,
                       help="If not empty, will start from the specified checkpoint (path to model###.pt file).")

def add_pretrained_VQ_options(parser):
    group = parser.add_argument_group('pretrained_VQ')
    # , default = './pre-trained/human-dog-ljy2'
    group.add_argument('--pretrained_save', type=str, required=True)
    group.add_argument('--plot_save', type=str, default='./results/plots')
    group.add_argument('--plot_cnt', type=int, default=5)

def train_args():
    parser = ArgumentParser()
    add_base_options(parser)
    add_data_options(parser)
    add_model_options(parser)
    add_diffusion_options(parser)
    add_training_options(parser)
    add_pretrained_VQ_options(parser)
    return parser.parse_args()

def get_model_args(args, vq_args, data):

    # default args
    clip_version = 'ViT-B/32'
    action_emb = 'tensor'
    cond_mode = get_cond_mode(args)
    if hasattr(data.dataset, 'num_actions'):
        num_actions = data.dataset.num_actions
    else:
        num_actions = 1

    # SMPL defaults
    print("Warning: Using Irregular code to init style size")
    data_rep = 'hml_vec'
    # print(type(data.dataset))
    if args.dataset == 'text2phase':
        nfeats = 0
        nrootfeats = 0
        ncond_feats = 512
        njoints = 0
        if getattr(args, 'use_manifold', False):
            # Manifold mode: diffuse continuous manifold(n_latent) + style(1) + traj(12)
            n_latent = vq_args.n_latent_channel
            nstylefeats = n_latent + 13  # manifold + style(1) + traj(12)
            ncontactfeats = nstylefeats
        else:
            # Legacy one-hot mode: diffuse one_hot(512) + angle(2) + style(1) + traj(12)
            ncontactfeats = 527
            nstylefeats = 527
    else:
        data_emd = data.dataset.dataset.feature_dim
        nfeats = 15
        nrootfeats = 12
        ncond_feats = vq_args.n_latent_channel
        ncontactfeats = 2
        njoints = int((data_emd-nrootfeats)/15)
        nstylefeats = 1


    return {'modeltype': '', 'njoints': njoints, 'nfeats': nfeats, 'nrootfeats': nrootfeats, 'nstylefeats': nstylefeats, 'ncond_feats': ncond_feats, 'ncontactfeats':ncontactfeats, 'num_actions': num_actions,
            'translation': True, 'pose_rep': 'rot6d', 'glob': True, 'glob_rot': True,
            'latent_dim': args.latent_dim, 'ff_size': 1024, 'num_layers': args.layers, 'num_heads': 4,
            'dropout': 0.1, 'activation': "gelu", 'data_rep': data_rep, 'cond_mode': cond_mode,
            'cond_mask_prob': args.cond_mask_prob, 'action_emb': action_emb, 'arch': args.arch,
            'emb_trans_dec': args.emb_trans_dec, 'clip_version': clip_version, 'dataset': args.dataset, 'generated_phase': getattr(args, 'generated_phase', False), 'use_manifold': getattr(args, 'use_manifold', False)}
def create_gaussian_diffusion(args):
    # default params
    predict_xstart = True  # we always predict x_start (a.k.a. x0), that's our deal!
    steps = args.diffusion_steps
    scale_beta = 1.  # no scaling
    timestep_respacing = ''  # can be used for ddim sampling, we don't use it.
    learn_sigma = False
    rescale_timesteps = False

    betas = gd.get_named_beta_schedule(args.noise_schedule, steps, scale_beta)
    loss_type = gd.LossType.MSE

    if not timestep_respacing:
        timestep_respacing = [steps]

    return SpacedDiffusion(
        use_timesteps=space_timesteps(steps, timestep_respacing),
        betas=betas,
        model_mean_type=(
            gd.ModelMeanType.EPSILON if not predict_xstart else gd.ModelMeanType.START_X
        ),
        model_var_type=(
            (
                gd.ModelVarType.FIXED_LARGE
                if not args.sigma_small
                else gd.ModelVarType.FIXED_SMALL
            )
            if not learn_sigma
            else gd.ModelVarType.LEARNED_RANGE
        ),
        loss_type=loss_type,
        rescale_timesteps=rescale_timesteps,
        lambda_vel=args.lambda_vel,
        lambda_rcxyz=args.lambda_rcxyz,
        lambda_fc=args.lambda_fc,
    )
def create_model_and_diffusion(args, vq_args, data):
    model = MDM(**get_model_args(args, vq_args, data))
    diffusion = create_gaussian_diffusion(args)
    return model, diffusion

if __name__ == '__main__':
    diff_args = train_args()
    print(torch.cuda.is_available())