# This code is based on https://github.com/openai/guided-diffusion
"""
This code started out as a PyTorch port of Ho et al's diffusion models:
https://github.com/hojonathanho/diffusion/blob/1e0dceb3b3495bbe19116a5e1b3596cd0706c543/diffusion_tf/diffusion_utils_2.py

Docstrings have been added, as well as DDIM sampling and a new collection of beta schedules.
"""

import enum
import math

import numpy as np
import torch
import torch as th
from copy import deepcopy
from diffusion.nn import mean_flat, sum_flat
from diffusion.losses import normal_kl, discretized_gaussian_log_likelihood
# from data_loaders.humanml.scripts import motion_process
import torch.nn.functional as F


def get_named_beta_schedule(schedule_name, num_diffusion_timesteps, scale_betas=1.):
    """
    Get a pre-defined beta schedule for the given name.

    The beta schedule library consists of beta schedules which remain similar
    in the limit of num_diffusion_timesteps.
    Beta schedules may be added, but should not be removed or changed once
    they are committed to maintain backwards compatibility.
    """
    if schedule_name == "linear":
        # Linear schedule from Ho et al, extended to work for any number of
        # diffusion steps.
        scale = scale_betas * 1000 / num_diffusion_timesteps
        beta_start = scale * 0.0001
        beta_end = scale * 0.02
        return np.linspace(
            beta_start, beta_end, num_diffusion_timesteps, dtype=np.float64
        )
    elif schedule_name == "cosine":
        return betas_for_alpha_bar(
            num_diffusion_timesteps,
            lambda t: math.cos((t + 0.008) / 1.008 * math.pi / 2) ** 2,
        )
    else:
        raise NotImplementedError(f"unknown beta schedule: {schedule_name}")


def betas_for_alpha_bar(num_diffusion_timesteps, alpha_bar, max_beta=0.999):
    """
    Create a beta schedule that discretizes the given alpha_t_bar function,
    which defines the cumulative product of (1-beta) over time from t = [0,1].

    :param num_diffusion_timesteps: the number of betas to produce.
    :param alpha_bar: a lambda that takes an argument t from 0 to 1 and
                      produces the cumulative product of (1-beta) up to that
                      part of the diffusion process.
    :param max_beta: the maximum beta to use; use values lower than 1 to
                     prevent singularities.
    """
    betas = []
    for i in range(num_diffusion_timesteps):
        t1 = i / num_diffusion_timesteps
        t2 = (i + 1) / num_diffusion_timesteps
        betas.append(min(1 - alpha_bar(t2) / alpha_bar(t1), max_beta))
    return np.array(betas)


class ModelMeanType(enum.Enum):
    """
    Which type of output the model predicts.
    """

    PREVIOUS_X = enum.auto()  # the model predicts x_{t-1}
    START_X = enum.auto()  # the model predicts x_0
    EPSILON = enum.auto()  # the model predicts epsilon


class ModelVarType(enum.Enum):
    """
    What is used as the model's output variance.

    The LEARNED_RANGE option has been added to allow the model to predict
    values between FIXED_SMALL and FIXED_LARGE, making its job easier.
    """

    LEARNED = enum.auto()
    FIXED_SMALL = enum.auto()
    FIXED_LARGE = enum.auto()
    LEARNED_RANGE = enum.auto()


class LossType(enum.Enum):
    MSE = enum.auto()  # use raw MSE loss (and KL when learning variances)
    RESCALED_MSE = (
        enum.auto()
    )  # use raw MSE loss (with RESCALED_KL when learning variances)
    KL = enum.auto()  # use the variational lower-bound
    RESCALED_KL = enum.auto()  # like KL, but rescale to estimate the full VLB

    def is_vb(self):
        return self == LossType.KL or self == LossType.RESCALED_KL


class GaussianDiffusion:
    """
    Utilities for training and sampling diffusion models.

    Ported directly from here, and then adapted over time to further experimentation.
    https://github.com/hojonathanho/diffusion/blob/1e0dceb3b3495bbe19116a5e1b3596cd0706c543/diffusion_tf/diffusion_utils_2.py#L42

    :param betas: a 1-D numpy array of betas for each diffusion timestep,
                  starting at T and going to 1.
    :param model_mean_type: a ModelMeanType determining what the model outputs.
    :param model_var_type: a ModelVarType determining how variance is output.
    :param loss_type: a LossType determining the loss function to use.
    :param rescale_timesteps: if True, pass floating point timesteps into the
                              model so that they are always scaled like in the
                              original paper (0 to 1000).
    """

    def __init__(
        self,
        *,
        betas,
        model_mean_type,
        model_var_type,
        loss_type,
        rescale_timesteps=False,
        lambda_rcxyz=0.,
        lambda_vel=0.,
        lambda_pose=1.,
        lambda_orient=1.,
        lambda_loc=1.,
        data_rep='rot6d',
        lambda_root_vel=0.,
        lambda_vel_rcxyz=0.,
        lambda_fc=0.,
    ):
        self.model_mean_type = model_mean_type
        self.model_var_type = model_var_type
        self.loss_type = loss_type
        self.rescale_timesteps = rescale_timesteps
        self.data_rep = data_rep

        if data_rep != 'rot_vel' and lambda_pose != 1.:
            raise ValueError('lambda_pose is relevant only when training on velocities!')
        self.lambda_pose = lambda_pose
        self.lambda_orient = lambda_orient
        self.lambda_loc = lambda_loc

        self.lambda_rcxyz = lambda_rcxyz
        self.lambda_vel = lambda_vel
        self.lambda_root_vel = lambda_root_vel
        self.lambda_vel_rcxyz = lambda_vel_rcxyz
        self.lambda_fc = lambda_fc

        if self.lambda_rcxyz > 0. or self.lambda_vel > 0. or self.lambda_root_vel > 0. or \
                self.lambda_vel_rcxyz > 0. or self.lambda_fc > 0.:
            assert self.loss_type == LossType.MSE, 'Geometric losses are supported by MSE loss type only!'

        # Use float64 for accuracy.
        betas = np.array(betas, dtype=np.float64)
        self.betas = betas
        assert len(betas.shape) == 1, "betas must be 1-D"
        assert (betas > 0).all() and (betas <= 1).all()

        self.num_timesteps = int(betas.shape[0])

        alphas = 1.0 - betas
        self.alphas_cumprod = np.cumprod(alphas, axis=0)
        self.alphas_cumprod_prev = np.append(1.0, self.alphas_cumprod[:-1])
        self.alphas_cumprod_next = np.append(self.alphas_cumprod[1:], 0.0)
        assert self.alphas_cumprod_prev.shape == (self.num_timesteps,)

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.sqrt_alphas_cumprod = np.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = np.sqrt(1.0 - self.alphas_cumprod)
        self.log_one_minus_alphas_cumprod = np.log(1.0 - self.alphas_cumprod)
        self.sqrt_recip_alphas_cumprod = np.sqrt(1.0 / self.alphas_cumprod)
        self.sqrt_recipm1_alphas_cumprod = np.sqrt(1.0 / self.alphas_cumprod - 1)

        # calculations for posterior q(x_{t-1} | x_t, x_0)
        self.posterior_variance = (
            betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        )
        # log calculation clipped because the posterior variance is 0 at the
        # beginning of the diffusion chain.
        self.posterior_log_variance_clipped = np.log(
            np.append(self.posterior_variance[1], self.posterior_variance[1:])
        )
        self.posterior_mean_coef1 = (
            betas * np.sqrt(self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        )
        self.posterior_mean_coef2 = (
            (1.0 - self.alphas_cumprod_prev)
            * np.sqrt(alphas)
            / (1.0 - self.alphas_cumprod)
        )

        self.l2_loss = lambda a, b: (a - b) ** 2  # th.nn.MSELoss(reduction='none')  # must be None for handling mask later on.

    def masked_l2(self, a, b, mask):
        # assuming a.shape == b.shape == bs, J, Jdim, seqlen
        # assuming mask.shape == bs, 1, 1, seqlen
        loss = self.l2_loss(a, b)
        loss = sum_flat(loss * mask.float())  # gives \sigma_euclidean over unmasked elements
        n_entries = a.shape[1] * a.shape[2]
        non_zero_elements = sum_flat(mask) * n_entries
        # print('mask', mask.shape)
        # print('non_zero_elements', non_zero_elements)
        # print('loss', loss)
        mse_loss_val = loss / non_zero_elements
        # print('mse_loss_val', mse_loss_val)
        return mse_loss_val


    def q_mean_variance(self, x_start, t):
        """
        Get the distribution q(x_t | x_0).

        :param x_start: the [N x C x ...] tensor of noiseless inputs.
        :param t: the number of diffusion steps (minus 1). Here, 0 means one step.
        :return: A tuple (mean, variance, log_variance), all of x_start's shape.
        """
        mean = (
            _extract_into_tensor(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start
        )
        variance = _extract_into_tensor(1.0 - self.alphas_cumprod, t, x_start.shape)
        log_variance = _extract_into_tensor(
            self.log_one_minus_alphas_cumprod, t, x_start.shape
        )
        return mean, variance, log_variance

    def q_sample(self, x_start, t, noise=None):
        """
        Diffuse the dataset for a given number of diffusion steps.

        In other words, sample from q(x_t | x_0).

        :param x_start: the initial dataset batch.
        :param t: the number of diffusion steps (minus 1). Here, 0 means one step.
        :param noise: if specified, the split-out normal noise.
        :return: A noisy version of x_start.
        """
        if noise is None:
            noise = th.randn_like(x_start)
        assert noise.shape == x_start.shape
        return (
            _extract_into_tensor(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start
            + _extract_into_tensor(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape)
            * noise
        )

    def q_posterior_mean_variance(self, x_start, x_t, t):
        """
        Compute the mean and variance of the diffusion posterior:

            q(x_{t-1} | x_t, x_0)

        """
        assert x_start.shape == x_t.shape
        posterior_mean = (
            _extract_into_tensor(self.posterior_mean_coef1, t, x_t.shape) * x_start
            + _extract_into_tensor(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_variance = _extract_into_tensor(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = _extract_into_tensor(
            self.posterior_log_variance_clipped, t, x_t.shape
        )
        assert (
            posterior_mean.shape[0]
            == posterior_variance.shape[0]
            == posterior_log_variance_clipped.shape[0]
            == x_start.shape[0]
        )
        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    def p_mean_variance(
        self, model, x, t, clip_denoised=True, denoised_fn=None, model_kwargs=None
    ):
        """
        Apply the model to get p(x_{t-1} | x_t), as well as a prediction of
        the initial x, x_0.

        :param model: the model, which takes a signal and a batch of timesteps
                      as input.
        :param x: the [N x C x ...] tensor at time t.
        :param t: a 1-D Tensor of timesteps.
        :param clip_denoised: if True, clip the denoised signal into [-1, 1].
        :param denoised_fn: if not None, a function which applies to the
            x_start prediction before it is used to sample. Applies before
            clip_denoised.
        :param model_kwargs: if not None, a dict of extra keyword arguments to
            pass to the model. This can be used for conditioning.
        :return: a dict with the following keys:
                 - 'mean': the model mean output.
                 - 'variance': the model variance output.
                 - 'log_variance': the log of 'variance'.
                 - 'pred_xstart': the prediction for x_0.
        """
        if model_kwargs is None:
            model_kwargs = {}

        B, C = x.shape[:2]
        assert t.shape == (B,)
        if model.model.dataset != 'text2phase':
            motion_cond = model_kwargs['y']['text_embed'].reshape(x.shape[0], -1, x.shape[2])
            stylecode = model_kwargs['y']['stylecode']
            relative2start_rootpos = model_kwargs['y']['relative2start_rootpos']
            x_t_withcond = torch.concat((x, stylecode, motion_cond, relative2start_rootpos), dim=1)

            # test Onnx file: 12-06-2025
            # import onnxruntime as ort
            # import numpy as np
            # # Load ONNX model
            # providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
            # ort_session = ort.InferenceSession("results/t100_p2m.onnx", providers=providers)

            # onnx_inputs = {
            #     "motion_noise": x.cpu().numpy(),
            #     "stylecode": stylecode.cpu().numpy(),
            #     "manifold": motion_cond.cpu().numpy(),
            #     "motion_root12": relative2start_rootpos.cpu().numpy(),
            #     "timesteps": t.cpu().numpy().astype(np.int64),  # explicitly specify int64 type
            # }
            model_output = model(x_t_withcond, self._scale_timesteps(t), **model_kwargs)
            split_sizes = [model_output.size(1) - 2, 2]
            model_output, footcontact_label = torch.split(model_output, split_sizes, dim=1)
            model_output = self.motion_postprocess_gen(model_output.permute(0, 2, 1)).permute(0, 2, 1)

            # test Onnx file: 12-06-2025
            # onnx_outputs = ort_session.run(None, onnx_inputs)
            # model_output_onnx = torch.tensor(onnx_outputs[0], device=x_t_withcond.device)
            # split_sizes = [model_output_onnx.size(1) - 2, 2]
            # model_output, footcontact_label = torch.split(model_output_onnx, split_sizes, dim=1)
            # model_output = self.motion_postprocess_gen(model_output.permute(0, 2, 1)).permute(0, 2, 1)
        else:
            x_t_withcond = x
            model_output = model(x_t_withcond, self._scale_timesteps(t), **model_kwargs)

            use_manifold = getattr(model, 'use_manifold', False)
            if hasattr(model, 'model'):
                use_manifold = getattr(model.model, 'use_manifold', use_manifold)

            if use_manifold:
                # Manifold mode: no softmax or special postprocessing needed
                # Just apply trajectory postprocess (rotation normalization)
                footcontact_label = None
                n_manifold = model_output.shape[1] - 13  # total - style(1) - traj(12)
                model_output = self.generated_phase_postprocess_manifold(model_output, n_manifold)
                mask_expanded = model_kwargs['y']['masks'].permute(0, 2, 1)
                mask_expanded = mask_expanded.expand(-1, model_output.shape[1], -1)
                model_output = model_output * mask_expanded
            else:
                # Legacy one-hot mode
                if t[0] != 0:
                    model_output[:, :512, :] = F.softmax(model_output[:, :512, :], dim=1)

                footcontact_label = None
                model_output = self.generated_phase_postprocess_gen(model_output)

                mask_expanded = model_kwargs['y']['masks'].permute(0, 2, 1)
                mask_expanded = mask_expanded.expand(-1, 527, -1)
                model_output = model_output * mask_expanded
                # we pad the one hot prediction with [1, 0, .. ]
                inv_mask = ~mask_expanded.bool()
                model_output[:, 0, :][inv_mask[:, 0, :]] = 1
            # model_output_numpy = model_output.cpu().numpy()

        if 'inpainting_mask' in model_kwargs['y'].keys() and 'inpainted_motion' in model_kwargs['y'].keys():
            inpainting_mask, inpainted_motion = model_kwargs['y']['inpainting_mask'], model_kwargs['y']['inpainted_motion']
            assert self.model_mean_type == ModelMeanType.START_X, 'This feature supports only X_start pred for mow!'
            assert model_output.shape == inpainting_mask.shape == inpainted_motion.shape
            model_output = (model_output * ~inpainting_mask) + (inpainted_motion * inpainting_mask)
            # print('model_output', model_output.shape, model_output)
            # print('inpainting_mask', inpainting_mask.shape, inpainting_mask[0,0,0,:])
            # print('inpainted_motion', inpainted_motion.shape, inpainted_motion)

        if self.model_var_type in [ModelVarType.LEARNED, ModelVarType.LEARNED_RANGE]:
            assert model_output.shape == (B, C * 2, *x.shape[2:])
            model_output, model_var_values = th.split(model_output, C, dim=1)
            if self.model_var_type == ModelVarType.LEARNED:
                model_log_variance = model_var_values
                model_variance = th.exp(model_log_variance)
            else:
                min_log = _extract_into_tensor(
                    self.posterior_log_variance_clipped, t, x.shape
                )
                max_log = _extract_into_tensor(np.log(self.betas), t, x.shape)
                # The model_var_values is [-1, 1] for [min_var, max_var].
                frac = (model_var_values + 1) / 2
                model_log_variance = frac * max_log + (1 - frac) * min_log
                model_variance = th.exp(model_log_variance)
        else:
            model_variance, model_log_variance = {
                # for fixedlarge, we set the initial (log-)variance like so
                # to get a better decoder log likelihood.
                ModelVarType.FIXED_LARGE: (
                    np.append(self.posterior_variance[1], self.betas[1:]),
                    np.log(np.append(self.posterior_variance[1], self.betas[1:])),
                ),
                ModelVarType.FIXED_SMALL: (
                    self.posterior_variance,
                    self.posterior_log_variance_clipped,
                ),
            }[self.model_var_type]
            # print('model_variance', model_variance)
            # print('model_log_variance',model_log_variance)
            # print('self.posterior_variance', self.posterior_variance)
            # print('self.posterior_log_variance_clipped', self.posterior_log_variance_clipped)
            # print('self.model_var_type', self.model_var_type)


            model_variance = _extract_into_tensor(model_variance, t, x.shape)
            model_log_variance = _extract_into_tensor(model_log_variance, t, x.shape)

        def process_xstart(x):
            if denoised_fn is not None:
                x = denoised_fn(x)
            if clip_denoised:
                # print('clip_denoised', clip_denoised)
                return x.clamp(-1, 1)
            return x

        if self.model_mean_type == ModelMeanType.PREVIOUS_X:
            pred_xstart = process_xstart(
                self._predict_xstart_from_xprev(x_t=x, t=t, xprev=model_output)
            )
            model_mean = model_output
        elif self.model_mean_type in [ModelMeanType.START_X, ModelMeanType.EPSILON]:  # THIS IS US!
            if self.model_mean_type == ModelMeanType.START_X:
                pred_xstart = process_xstart(model_output)
            else:
                pred_xstart = process_xstart(
                    self._predict_xstart_from_eps(x_t=x, t=t, eps=model_output)
                )
            model_mean, _, _ = self.q_posterior_mean_variance(
                x_start=pred_xstart, x_t=x, t=t
            )
        else:
            raise NotImplementedError(self.model_mean_type)

        assert (
            model_mean.shape == model_log_variance.shape == pred_xstart.shape == x.shape
        )
        return {
            "mean": model_mean,
            "variance": model_variance,
            "log_variance": model_log_variance,
            "pred_xstart": pred_xstart,
            "contact_label": footcontact_label
        }

    def _predict_xstart_from_eps(self, x_t, t, eps):
        assert x_t.shape == eps.shape
        return (
            _extract_into_tensor(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t
            - _extract_into_tensor(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * eps
        )

    def _predict_xstart_from_xprev(self, x_t, t, xprev):
        assert x_t.shape == xprev.shape
        return (  # (xprev - coef2*x_t) / coef1
            _extract_into_tensor(1.0 / self.posterior_mean_coef1, t, x_t.shape) * xprev
            - _extract_into_tensor(
                self.posterior_mean_coef2 / self.posterior_mean_coef1, t, x_t.shape
            )
            * x_t
        )

    def _predict_eps_from_xstart(self, x_t, t, pred_xstart):
        return (
            _extract_into_tensor(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t
            - pred_xstart
        ) / _extract_into_tensor(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape)

    def _scale_timesteps(self, t):
        if self.rescale_timesteps:
            return t.float() * (1000.0 / self.num_timesteps)
        return t

    def condition_mean(self, cond_fn, p_mean_var, x, t, model_kwargs=None):
        """
        Compute the mean for the previous step, given a function cond_fn that
        computes the gradient of a conditional log probability with respect to
        x. In particular, cond_fn computes grad(log(p(y|x))), and we want to
        condition on y.

        This uses the conditioning strategy from Sohl-Dickstein et al. (2015).
        """
        gradient = cond_fn(x, self._scale_timesteps(t), **model_kwargs)
        new_mean = (
            p_mean_var["mean"].float() + p_mean_var["variance"] * gradient.float()
        )
        return new_mean

    def condition_mean_with_grad(self, cond_fn, p_mean_var, x, t, model_kwargs=None):
        """
        Compute the mean for the previous step, given a function cond_fn that
        computes the gradient of a conditional log probability with respect to
        x. In particular, cond_fn computes grad(log(p(y|x))), and we want to
        condition on y.

        This uses the conditioning strategy from Sohl-Dickstein et al. (2015).
        """
        gradient = cond_fn(x, t, p_mean_var, **model_kwargs)
        new_mean = (
            p_mean_var["mean"].float() + p_mean_var["variance"] * gradient.float()
        )
        return new_mean

    def condition_score(self, cond_fn, p_mean_var, x, t, model_kwargs=None):
        """
        Compute what the p_mean_variance output would have been, should the
        model's score function be conditioned by cond_fn.

        See condition_mean() for details on cond_fn.

        Unlike condition_mean(), this instead uses the conditioning strategy
        from Song et al (2020).
        """
        alpha_bar = _extract_into_tensor(self.alphas_cumprod, t, x.shape)

        eps = self._predict_eps_from_xstart(x, t, p_mean_var["pred_xstart"])
        eps = eps - (1 - alpha_bar).sqrt() * cond_fn(
            x, self._scale_timesteps(t), **model_kwargs
        )

        out = p_mean_var.copy()
        out["pred_xstart"] = self._predict_xstart_from_eps(x, t, eps)
        out["mean"], _, _ = self.q_posterior_mean_variance(
            x_start=out["pred_xstart"], x_t=x, t=t
        )
        return out

    def condition_score_with_grad(self, cond_fn, p_mean_var, x, t, model_kwargs=None):
        """
        Compute what the p_mean_variance output would have been, should the
        model's score function be conditioned by cond_fn.

        See condition_mean() for details on cond_fn.

        Unlike condition_mean(), this instead uses the conditioning strategy
        from Song et al (2020).
        """
        alpha_bar = _extract_into_tensor(self.alphas_cumprod, t, x.shape)

        eps = self._predict_eps_from_xstart(x, t, p_mean_var["pred_xstart"])
        eps = eps - (1 - alpha_bar).sqrt() * cond_fn(
            x, t, p_mean_var, **model_kwargs
        )

        out = p_mean_var.copy()
        out["pred_xstart"] = self._predict_xstart_from_eps(x, t, eps)
        out["mean"], _, _ = self.q_posterior_mean_variance(
            x_start=out["pred_xstart"], x_t=x, t=t
        )
        return out

    def p_sample(
        self,
        model,
        x,
        t,
        clip_denoised=True,
        denoised_fn=None,
        cond_fn=None,
        model_kwargs=None,
        const_noise=False,
    ):
        """
        Sample x_{t-1} from the model at the given timestep.

        :param model: the model to sample from.
        :param x: the current tensor at x_{t-1}.
        :param t: the value of t, starting at 0 for the first diffusion step.
        :param clip_denoised: if True, clip the x_start prediction to [-1, 1].
        :param denoised_fn: if not None, a function which applies to the
            x_start prediction before it is used to sample.
        :param cond_fn: if not None, this is a gradient function that acts
                        similarly to the model.
        :param model_kwargs: if not None, a dict of extra keyword arguments to
            pass to the model. This can be used for conditioning.
        :return: a dict containing the following keys:
                 - 'sample': a random sample from the model.
                 - 'pred_xstart': a prediction of x_0.
        """
        out = self.p_mean_variance(
            model,
            x,
            t,
            clip_denoised=clip_denoised,
            denoised_fn=denoised_fn,
            model_kwargs=model_kwargs,
        )
        noise = th.randn_like(x)
        # print('const_noise', const_noise)
        if const_noise:
            noise = noise[[0]].repeat(x.shape[0], 1, 1, 1)

        nonzero_mask = (
            (t != 0).float().view(-1, *([1] * (len(x.shape) - 1)))
        )  # no noise when t == 0
        if cond_fn is not None:
            out["mean"] = self.condition_mean(
                cond_fn, out, x, t, model_kwargs=model_kwargs
            )
        # print('mean', out["mean"].shape, out["mean"])
        # print('log_variance', out["log_variance"].shape, out["log_variance"])
        # print('nonzero_mask', nonzero_mask.shape, nonzero_mask)
        sample = out["mean"] + nonzero_mask * th.exp(0.5 * out["log_variance"]) * noise
        return {"sample": sample, "pred_xstart": out["pred_xstart"], "contact_label": out["contact_label"]}

    def p_sample_with_grad(
        self,
        model,
        x,
        t,
        clip_denoised=True,
        denoised_fn=None,
        cond_fn=None,
        model_kwargs=None,
    ):
        """
        Sample x_{t-1} from the model at the given timestep.

        :param model: the model to sample from.
        :param x: the current tensor at x_{t-1}.
        :param t: the value of t, starting at 0 for the first diffusion step.
        :param clip_denoised: if True, clip the x_start prediction to [-1, 1].
        :param denoised_fn: if not None, a function which applies to the
            x_start prediction before it is used to sample.
        :param cond_fn: if not None, this is a gradient function that acts
                        similarly to the model.
        :param model_kwargs: if not None, a dict of extra keyword arguments to
            pass to the model. This can be used for conditioning.
        :return: a dict containing the following keys:
                 - 'sample': a random sample from the model.
                 - 'pred_xstart': a prediction of x_0.
        """
        with th.enable_grad():
            x = x.detach().requires_grad_()
            out = self.p_mean_variance(
                model,
                x,
                t,
                clip_denoised=clip_denoised,
                denoised_fn=denoised_fn,
                model_kwargs=model_kwargs,
            )
            noise = th.randn_like(x)
            nonzero_mask = (
                (t != 0).float().view(-1, *([1] * (len(x.shape) - 1)))
            )  # no noise when t == 0
            if cond_fn is not None:
                out["mean"] = self.condition_mean_with_grad(
                    cond_fn, out, x, t, model_kwargs=model_kwargs
                )
        sample = out["mean"] + nonzero_mask * th.exp(0.5 * out["log_variance"]) * noise
        return {"sample": sample, "pred_xstart": out["pred_xstart"].detach()}

    def p_sample_loop(
        self,
        model,
        shape,
        noise=None,
        clip_denoised=True,
        denoised_fn=None,
        cond_fn=None,
        model_kwargs=None,
        device=None,
        progress=False,
        skip_timesteps=0,
        init_image=None,
        randomize_class=False,
        cond_fn_with_grad=False,
        dump_steps=None,
        const_noise=False,
    ):
        """
        Generate samples from the model.

        :param model: the model module.
        :param shape: the shape of the samples, (N, C, H, W).
        :param noise: if specified, the noise from the encoder to sample.
                      Should be of the same shape as `shape`.
        :param clip_denoised: if True, clip x_start predictions to [-1, 1].
        :param denoised_fn: if not None, a function which applies to the
            x_start prediction before it is used to sample.
        :param cond_fn: if not None, this is a gradient function that acts
                        similarly to the model.
        :param model_kwargs: if not None, a dict of extra keyword arguments to
            pass to the model. This can be used for conditioning.
        :param device: if specified, the device to create the samples on.
                       If not specified, use a model parameter's device.
        :param progress: if True, show a tqdm progress bar.
        :param const_noise: If True, will noise all samples with the same noise throughout sampling
        :return: a non-differentiable batch of samples.
        """
        final = None
        if dump_steps is not None:
            dump = []

        if 'text' in model_kwargs['y'].keys():
            # encoding once instead of each iteration saves lots of time
            model_kwargs['y']['text_embed'] = model.encode_text(model_kwargs['y']['text'])

        motion_record = []
        for i, sample in enumerate(self.p_sample_loop_progressive(
            model,
            shape,
            noise=noise,
            clip_denoised=clip_denoised,
            denoised_fn=denoised_fn,
            cond_fn=cond_fn,
            model_kwargs=model_kwargs,
            device=device,
            progress=progress,
            skip_timesteps=skip_timesteps,
            init_image=init_image,
            randomize_class=randomize_class,
            cond_fn_with_grad=cond_fn_with_grad,
            const_noise=const_noise,
        )):
            if dump_steps is not None and i in dump_steps:
                dump.append(deepcopy(sample["sample"]))
            if i % 100 == 0:
                motion_record.append(sample)
            final = sample
        if dump_steps is not None:
            return dump
        return final["sample"], motion_record, final["contact_label"]

    def p_sample_loop_progressive(
        self,
        model,
        shape,
        noise=None,
        clip_denoised=True,
        denoised_fn=None,
        cond_fn=None,
        model_kwargs=None,
        device=None,
        progress=False,
        skip_timesteps=0,
        init_image=None,
        randomize_class=False,
        cond_fn_with_grad=False,
        const_noise=False,
    ):
        """
        Generate samples from the model and yield intermediate samples from
        each timestep of diffusion.

        Arguments are the same as p_sample_loop().
        Returns a generator over dicts, where each dict is the return value of
        p_sample().
        """
        if device is None:
            device = next(model.parameters()).device
        assert isinstance(shape, (tuple, list))
        if noise is not None:
            img = noise
        else:
            img = th.randn(*shape, device=device)

        if skip_timesteps and init_image is None:
            init_image = th.zeros_like(img)

        indices = list(range(self.num_timesteps - skip_timesteps))[::-1]

        if init_image is not None:
            my_t = th.ones([shape[0]], device=device, dtype=th.long) * indices[0]
            img = self.q_sample(init_image, my_t, img)

        if progress:
            # Lazy import so that we don't depend on tqdm.
            from tqdm.auto import tqdm

            indices = tqdm(indices)

        for i in indices:
            t = th.tensor([i] * shape[0], device=device)
            if randomize_class and 'y' in model_kwargs:
                model_kwargs['y'] = th.randint(low=0, high=model.num_classes,
                                               size=model_kwargs['y'].shape,
                                               device=model_kwargs['y'].device)
            with th.no_grad():
                # sample_fn = self.p_sample_with_grad if cond_fn_with_grad else self.p_sample
                if cond_fn_with_grad:
                    out = self.p_sample_with_grad(
                        model,
                        img,
                        t,
                        clip_denoised=clip_denoised,
                        denoised_fn=denoised_fn,
                        cond_fn=cond_fn,
                        model_kwargs=model_kwargs,
                    )
                else:
                    out = self.p_sample(
                        model,
                        img,
                        t,
                        clip_denoised=clip_denoised,
                        denoised_fn=denoised_fn,
                        cond_fn=cond_fn,
                        model_kwargs=model_kwargs,
                        const_noise=const_noise,
                    )
                yield out
                img = out["sample"]

    def ddim_sample(
        self,
        model,
        x,
        t,
        clip_denoised=True,
        denoised_fn=None,
        cond_fn=None,
        model_kwargs=None,
        eta=0.0,
    ):
        """
        Sample x_{t-1} from the model using DDIM.

        Same usage as p_sample().
        """
        out_orig = self.p_mean_variance(
            model,
            x,
            t,
            clip_denoised=clip_denoised,
            denoised_fn=denoised_fn,
            model_kwargs=model_kwargs,
        )
        if cond_fn is not None:
            out = self.condition_score(cond_fn, out_orig, x, t, model_kwargs=model_kwargs)
        else:
            out = out_orig

        # Usually our model outputs epsilon, but we re-derive it
        # in case we used x_start or x_prev prediction.
        eps = self._predict_eps_from_xstart(x, t, out["pred_xstart"])

        alpha_bar = _extract_into_tensor(self.alphas_cumprod, t, x.shape)
        alpha_bar_prev = _extract_into_tensor(self.alphas_cumprod_prev, t, x.shape)
        sigma = (
            eta
            * th.sqrt((1 - alpha_bar_prev) / (1 - alpha_bar))
            * th.sqrt(1 - alpha_bar / alpha_bar_prev)
        )
        # Equation 12.
        noise = th.randn_like(x)
        mean_pred = (
            out["pred_xstart"] * th.sqrt(alpha_bar_prev)
            + th.sqrt(1 - alpha_bar_prev - sigma ** 2) * eps
        )
        nonzero_mask = (
            (t != 0).float().view(-1, *([1] * (len(x.shape) - 1)))
        )  # no noise when t == 0
        sample = mean_pred + nonzero_mask * sigma * noise
        return {"sample": sample, "pred_xstart": out_orig["pred_xstart"]}

    def ddim_sample_with_grad(
        self,
        model,
        x,
        t,
        clip_denoised=True,
        denoised_fn=None,
        cond_fn=None,
        model_kwargs=None,
        eta=0.0,
    ):
        """
        Sample x_{t-1} from the model using DDIM.

        Same usage as p_sample().
        """
        with th.enable_grad():
            x = x.detach().requires_grad_()
            out_orig = self.p_mean_variance(
                model,
                x,
                t,
                clip_denoised=clip_denoised,
                denoised_fn=denoised_fn,
                model_kwargs=model_kwargs,
            )
            if cond_fn is not None:
                out = self.condition_score_with_grad(cond_fn, out_orig, x, t,
                                                     model_kwargs=model_kwargs)
            else:
                out = out_orig

        out["pred_xstart"] = out["pred_xstart"].detach()

        # Usually our model outputs epsilon, but we re-derive it
        # in case we used x_start or x_prev prediction.
        eps = self._predict_eps_from_xstart(x, t, out["pred_xstart"])

        alpha_bar = _extract_into_tensor(self.alphas_cumprod, t, x.shape)
        alpha_bar_prev = _extract_into_tensor(self.alphas_cumprod_prev, t, x.shape)
        sigma = (
            eta
            * th.sqrt((1 - alpha_bar_prev) / (1 - alpha_bar))
            * th.sqrt(1 - alpha_bar / alpha_bar_prev)
        )
        # Equation 12.
        noise = th.randn_like(x)
        mean_pred = (
            out["pred_xstart"] * th.sqrt(alpha_bar_prev)
            + th.sqrt(1 - alpha_bar_prev - sigma ** 2) * eps
        )
        nonzero_mask = (
            (t != 0).float().view(-1, *([1] * (len(x.shape) - 1)))
        )  # no noise when t == 0
        sample = mean_pred + nonzero_mask * sigma * noise
        return {"sample": sample, "pred_xstart": out_orig["pred_xstart"].detach()}

    def ddim_reverse_sample(
        self,
        model,
        x,
        t,
        clip_denoised=True,
        denoised_fn=None,
        model_kwargs=None,
        eta=0.0,
    ):
        """
        Sample x_{t+1} from the model using DDIM reverse ODE.
        """
        assert eta == 0.0, "Reverse ODE only for deterministic path"
        out = self.p_mean_variance(
            model,
            x,
            t,
            clip_denoised=clip_denoised,
            denoised_fn=denoised_fn,
            model_kwargs=model_kwargs,
        )
        # Usually our model outputs epsilon, but we re-derive it
        # in case we used x_start or x_prev prediction.
        eps = (
            _extract_into_tensor(self.sqrt_recip_alphas_cumprod, t, x.shape) * x
            - out["pred_xstart"]
        ) / _extract_into_tensor(self.sqrt_recipm1_alphas_cumprod, t, x.shape)
        alpha_bar_next = _extract_into_tensor(self.alphas_cumprod_next, t, x.shape)

        # Equation 12. reversed
        mean_pred = (
            out["pred_xstart"] * th.sqrt(alpha_bar_next)
            + th.sqrt(1 - alpha_bar_next) * eps
        )

        return {"sample": mean_pred, "pred_xstart": out["pred_xstart"]}

    def ddim_sample_loop(
        self,
        model,
        shape,
        noise=None,
        clip_denoised=True,
        denoised_fn=None,
        cond_fn=None,
        model_kwargs=None,
        device=None,
        progress=False,
        eta=0.0,
        skip_timesteps=0,
        init_image=None,
        randomize_class=False,
        cond_fn_with_grad=False,
        dump_steps=None,
        const_noise=False,
    ):
        """
        Generate samples from the model using DDIM.

        Same usage as p_sample_loop().
        """
        if dump_steps is not None:
            raise NotImplementedError()
        if const_noise == True:
            raise NotImplementedError()

        final = None
        for sample in self.ddim_sample_loop_progressive(
            model,
            shape,
            noise=noise,
            clip_denoised=clip_denoised,
            denoised_fn=denoised_fn,
            cond_fn=cond_fn,
            model_kwargs=model_kwargs,
            device=device,
            progress=progress,
            eta=eta,
            skip_timesteps=skip_timesteps,
            init_image=init_image,
            randomize_class=randomize_class,
            cond_fn_with_grad=cond_fn_with_grad,
        ):
            final = sample
        return final["sample"]

    def ddim_sample_loop_progressive(
        self,
        model,
        shape,
        noise=None,
        clip_denoised=True,
        denoised_fn=None,
        cond_fn=None,
        model_kwargs=None,
        device=None,
        progress=False,
        eta=0.0,
        skip_timesteps=0,
        init_image=None,
        randomize_class=False,
        cond_fn_with_grad=False,
    ):
        """
        Use DDIM to sample from the model and yield intermediate samples from
        each timestep of DDIM.

        Same usage as p_sample_loop_progressive().
        """
        if device is None:
            device = next(model.parameters()).device
        assert isinstance(shape, (tuple, list))
        if noise is not None:
            img = noise
        else:
            img = th.randn(*shape, device=device)

        if skip_timesteps and init_image is None:
            init_image = th.zeros_like(img)

        indices = list(range(self.num_timesteps - skip_timesteps))[::-1]

        if init_image is not None:
            my_t = th.ones([shape[0]], device=device, dtype=th.long) * indices[0]
            img = self.q_sample(init_image, my_t, img)

        if progress:
            # Lazy import so that we don't depend on tqdm.
            from tqdm.auto import tqdm

            indices = tqdm(indices)

        for i in indices:
            t = th.tensor([i] * shape[0], device=device)
            if randomize_class and 'y' in model_kwargs:
                model_kwargs['y'] = th.randint(low=0, high=model.num_classes,
                                               size=model_kwargs['y'].shape,
                                               device=model_kwargs['y'].device)
            with th.no_grad():
                sample_fn = self.ddim_sample_with_grad if cond_fn_with_grad else self.ddim_sample
                out = sample_fn(
                    model,
                    img,
                    t,
                    clip_denoised=clip_denoised,
                    denoised_fn=denoised_fn,
                    cond_fn=cond_fn,
                    model_kwargs=model_kwargs,
                    eta=eta,
                )
                yield out
                img = out["sample"]

    def plms_sample(
        self,
        model,
        x,
        t,
        clip_denoised=True,
        denoised_fn=None,
        cond_fn=None,
        model_kwargs=None,
        cond_fn_with_grad=False,
        order=2,
        old_out=None,
    ):
        """
        Sample x_{t-1} from the model using Pseudo Linear Multistep.

        Same usage as p_sample().
        """
        if not int(order) or not 1 <= order <= 4:
            raise ValueError('order is invalid (should be int from 1-4).')

        def get_model_output(x, t):
            with th.set_grad_enabled(cond_fn_with_grad and cond_fn is not None):
                x = x.detach().requires_grad_() if cond_fn_with_grad else x
                out_orig = self.p_mean_variance(
                    model,
                    x,
                    t,
                    clip_denoised=clip_denoised,
                    denoised_fn=denoised_fn,
                    model_kwargs=model_kwargs,
                )
                if cond_fn is not None:
                    if cond_fn_with_grad:
                        out = self.condition_score_with_grad(cond_fn, out_orig, x, t, model_kwargs=model_kwargs)
                        x = x.detach()
                    else:
                        out = self.condition_score(cond_fn, out_orig, x, t, model_kwargs=model_kwargs)
                else:
                    out = out_orig

            # Usually our model outputs epsilon, but we re-derive it
            # in case we used x_start or x_prev prediction.
            eps = self._predict_eps_from_xstart(x, t, out["pred_xstart"])
            return eps, out, out_orig

        alpha_bar = _extract_into_tensor(self.alphas_cumprod, t, x.shape)
        alpha_bar_prev = _extract_into_tensor(self.alphas_cumprod_prev, t, x.shape)
        eps, out, out_orig = get_model_output(x, t)

        if order > 1 and old_out is None:
            # Pseudo Improved Euler
            old_eps = [eps]
            mean_pred = out["pred_xstart"] * th.sqrt(alpha_bar_prev) + th.sqrt(1 - alpha_bar_prev) * eps
            eps_2, _, _ = get_model_output(mean_pred, t - 1)
            eps_prime = (eps + eps_2) / 2
            pred_prime = self._predict_xstart_from_eps(x, t, eps_prime)
            mean_pred = pred_prime * th.sqrt(alpha_bar_prev) + th.sqrt(1 - alpha_bar_prev) * eps_prime
        else:
            # Pseudo Linear Multistep (Adams-Bashforth)
            old_eps = old_out["old_eps"]
            old_eps.append(eps)
            cur_order = min(order, len(old_eps))
            if cur_order == 1:
                eps_prime = old_eps[-1]
            elif cur_order == 2:
                eps_prime = (3 * old_eps[-1] - old_eps[-2]) / 2
            elif cur_order == 3:
                eps_prime = (23 * old_eps[-1] - 16 * old_eps[-2] + 5 * old_eps[-3]) / 12
            elif cur_order == 4:
                eps_prime = (55 * old_eps[-1] - 59 * old_eps[-2] + 37 * old_eps[-3] - 9 * old_eps[-4]) / 24
            else:
                raise RuntimeError('cur_order is invalid.')
            pred_prime = self._predict_xstart_from_eps(x, t, eps_prime)
            mean_pred = pred_prime * th.sqrt(alpha_bar_prev) + th.sqrt(1 - alpha_bar_prev) * eps_prime

        if len(old_eps) >= order:
            old_eps.pop(0)

        nonzero_mask = (t != 0).float().view(-1, *([1] * (len(x.shape) - 1)))
        sample = mean_pred * nonzero_mask + out["pred_xstart"] * (1 - nonzero_mask)

        return {"sample": sample, "pred_xstart": out_orig["pred_xstart"], "old_eps": old_eps}

    def plms_sample_loop(
        self,
        model,
        shape,
        noise=None,
        clip_denoised=True,
        denoised_fn=None,
        cond_fn=None,
        model_kwargs=None,
        device=None,
        progress=False,
        skip_timesteps=0,
        init_image=None,
        randomize_class=False,
        cond_fn_with_grad=False,
        order=2,
    ):
        """
        Generate samples from the model using Pseudo Linear Multistep.

        Same usage as p_sample_loop().
        """
        final = None
        for sample in self.plms_sample_loop_progressive(
            model,
            shape,
            noise=noise,
            clip_denoised=clip_denoised,
            denoised_fn=denoised_fn,
            cond_fn=cond_fn,
            model_kwargs=model_kwargs,
            device=device,
            progress=progress,
            skip_timesteps=skip_timesteps,
            init_image=init_image,
            randomize_class=randomize_class,
            cond_fn_with_grad=cond_fn_with_grad,
            order=order,
        ):
            final = sample
        return final["sample"]

    def plms_sample_loop_progressive(
        self,
        model,
        shape,
        noise=None,
        clip_denoised=True,
        denoised_fn=None,
        cond_fn=None,
        model_kwargs=None,
        device=None,
        progress=False,
        skip_timesteps=0,
        init_image=None,
        randomize_class=False,
        cond_fn_with_grad=False,
        order=2,
    ):
        """
        Use PLMS to sample from the model and yield intermediate samples from each
        timestep of PLMS.

        Same usage as p_sample_loop_progressive().
        """
        if device is None:
            device = next(model.parameters()).device
        assert isinstance(shape, (tuple, list))
        if noise is not None:
            img = noise
        else:
            img = th.randn(*shape, device=device)

        if skip_timesteps and init_image is None:
            init_image = th.zeros_like(img)

        indices = list(range(self.num_timesteps - skip_timesteps))[::-1]

        if init_image is not None:
            my_t = th.ones([shape[0]], device=device, dtype=th.long) * indices[0]
            img = self.q_sample(init_image, my_t, img)

        if progress:
            # Lazy import so that we don't depend on tqdm.
            from tqdm.auto import tqdm

            indices = tqdm(indices)

        old_out = None

        for i in indices:
            t = th.tensor([i] * shape[0], device=device)
            if randomize_class and 'y' in model_kwargs:
                model_kwargs['y'] = th.randint(low=0, high=model.num_classes,
                                               size=model_kwargs['y'].shape,
                                               device=model_kwargs['y'].device)
            with th.no_grad():
                out = self.plms_sample(
                    model,
                    img,
                    t,
                    clip_denoised=clip_denoised,
                    denoised_fn=denoised_fn,
                    cond_fn=cond_fn,
                    model_kwargs=model_kwargs,
                    cond_fn_with_grad=cond_fn_with_grad,
                    order=order,
                    old_out=old_out,
                )
                yield out
                old_out = out
                img = out["sample"]

    def _vb_terms_bpd(
        self, model, x_start, x_t, t, clip_denoised=True, model_kwargs=None
    ):
        """
        Get a term for the variational lower-bound.

        The resulting units are bits (rather than nats, as one might expect).
        This allows for comparison to other papers.

        :return: a dict with the following keys:
                 - 'output': a shape [N] tensor of NLLs or KLs.
                 - 'pred_xstart': the x_0 predictions.
        """
        true_mean, _, true_log_variance_clipped = self.q_posterior_mean_variance(
            x_start=x_start, x_t=x_t, t=t
        )
        out = self.p_mean_variance(
            model, x_t, t, clip_denoised=clip_denoised, model_kwargs=model_kwargs
        )
        kl = normal_kl(
            true_mean, true_log_variance_clipped, out["mean"], out["log_variance"]
        )
        kl = mean_flat(kl) / np.log(2.0)

        decoder_nll = -discretized_gaussian_log_likelihood(
            x_start, means=out["mean"], log_scales=0.5 * out["log_variance"]
        )
        assert decoder_nll.shape == x_start.shape
        decoder_nll = mean_flat(decoder_nll) / np.log(2.0)

        # At the first timestep return the decoder NLL,
        # otherwise return KL(q(x_{t-1}|x_t,x_0) || p(x_{t-1}|x_t))
        output = th.where((t == 0), decoder_nll, kl)
        return {"output": output, "pred_xstart": out["pred_xstart"]}

    def transfer2absolute_batch(self, motion, init_root_position=None, init_rotation=None):
        assert motion.size(-1) in [417, 432, 582], "The last dimension of df_motion must be 432."
        device = motion.device
        batch_size, seq_len, _ = motion.shape

        if init_root_position is None:
            init_root_position = torch.zeros(batch_size, 3, device=device)
        if init_rotation is None:
            init_rotation = torch.eye(3, device=device).unsqueeze(0).repeat(batch_size, 1, 1)

        assert isinstance(motion, torch.Tensor), "motion must be a torch.Tensor"
        assert isinstance(init_root_position, torch.Tensor), "init_root_position must be a torch.Tensor"
        assert isinstance(init_rotation, torch.Tensor), "init_rotation must be a torch.Tensor"

        # Extract relative velocity and relative rotation matrix
        relative_velocity_local = motion[:, :, -12:-9]  # (batch_size, seq_len, 3)
        relative_rotations = motion[:, :, -9:].reshape(batch_size, seq_len, 3, 3)  # (batch_size, seq_len, 3, 3)

        # Compute the per-frame angle change
        cos_theta = relative_rotations[:, :, 0, 0]
        sin_theta = relative_rotations[:, :, 0, 2]
        angles_rad = torch.atan2(sin_theta, cos_theta)  # (batch_size, seq_len)

        # Initialize the first-frame angle
        init_cos_theta = init_rotation[:, 0, 0]  # (batch_size,)
        init_sin_theta = init_rotation[:, 0, 2]  # (batch_size,)
        first_angle_rad = torch.atan2(init_sin_theta, init_cos_theta)  # (batch_size,)
        angles_rad[:, 0] = first_angle_rad

        # Accumulate angle changes to obtain global angles
        absolute_angles_rad = torch.cumsum(angles_rad, dim=1)  # (batch_size, seq_len)

        def rotation_matrix_y(angle):
            """Generate a rotation matrix around the Y axis from an angle"""
            cos_theta = torch.cos(angle)
            sin_theta = torch.sin(angle)
            zeros = torch.zeros_like(cos_theta)
            ones = torch.ones_like(cos_theta)
            return torch.stack([
                torch.stack([cos_theta, zeros, sin_theta], dim=-1),
                torch.stack([zeros, ones, zeros], dim=-1),
                torch.stack([-sin_theta, zeros, cos_theta], dim=-1)
            ], dim=-2)

        # Generate absolute rotation matrices
        absolute_rotations = rotation_matrix_y(absolute_angles_rad)  # (batch_size, seq_len, 3, 3)

        # Initialize the absolute-position list
        absolute_positions = torch.zeros(batch_size, seq_len, 3, device=device)
        absolute_positions[:, 0, :] = init_root_position
        # Iterate over frames to compute absolute positions
        for t in range(1, seq_len):
            velocity_global = torch.bmm(absolute_rotations[:, t - 1],
                                        relative_velocity_local[:, t].unsqueeze(-1)).squeeze(-1)
            absolute_positions[:, t] = absolute_positions[:, t - 1] + velocity_global

        # Write absolute positions and rotation matrices back to motion
        updated_motion = motion.clone()
        updated_motion[:, :, -12:-9] = absolute_positions
        updated_motion[:, :, -9:] = absolute_rotations.reshape(batch_size, seq_len, -1)
        return updated_motion

    def integrate_root_motion(self, global_pos):
        """
        Integrate root-module information into the motion tensor.
        Convert positions and rotations to the root local coordinate system.

        :param global_pos: original motion tensor, shape is (batch_size, frames, 432)
        :return: integrated motion tensor, shape is (batch_size, frames, 420)
        """
        batch_size, frames, _ = global_pos.shape

        # Decompose the original tensor
        velocities = global_pos[..., :81]  # the first 84 dims are velocity
        positions = global_pos[..., 81:162]  # the middle 84 dims are position
        rotations = global_pos[..., 162:405]  # the next 252 dims are rotation
        root_pos_rot = global_pos[..., 405:]  # the last 12 dimensions are the root module: 3D position + 9D rotation matrix

        # Decompose root-module information
        root_positions = root_pos_rot[..., :3]  # root position is the first 3 dims
        root_rotations = root_pos_rot[..., 3:].reshape(batch_size, frames, 3, 3)  # root rotation matrix is 3x3
        # root_rotations_debug = root_pos_rot[..., 3:].numpy()

        # Transform positions to the root coordinate system
        # Ensure positions and root_positions dimensions match
        # positions_debug = positions.numpy()
        # rotations_debug = rotations.numpy()
        positions = positions.view(batch_size, frames, 27, 3)  # (batch_size, frames, 28, 3)
        root_positions = root_positions.unsqueeze(2)  # (batch_size, frames, 1, 3)

        # Transform positions to the root coordinate system
        positions_from_root = torch.matmul(
            positions, root_rotations.permute(0, 1, 3, 2)
        ) + root_positions
        # positions_from_root_debug = positions_from_root.numpy()

        # Transform rotations to the root coordinate system
        rotations = rotations.view(batch_size, frames, 27, 3, 3)  # rotation matrices for 28 joints per frame
        rotations_from_root = torch.matmul(
            root_rotations.unsqueeze(2), rotations
        )  # rotation matrices from world coordinates to root coordinates

        # Restore shape
        adjusted_positions = positions_from_root.view(batch_size, frames, -1)  # (batch_size, frames, 84)
        adjusted_rotations = rotations_from_root.view(batch_size, frames, -1)  # (batch_size, frames, 252)

        # Combine all parts
        updated_motion = torch.cat((velocities, adjusted_positions, adjusted_rotations), dim=-1)  # combine into 420 dims

        return updated_motion

    def motion_postprocess_train(self, df_motion):
        # Ensure the last dimension of motion is 432
        assert df_motion.size(-1) in {434, 432, 417,
                                      419, 582, 584}, "The last dimension of df_motion must be 434, 432, 417, or 419."
        root_start_index = 0
        if df_motion.size(-1) in {432, 434}:
            root_start_index = 420
        elif df_motion.size(-1) in {417, 419}:
            root_start_index = 405
        elif df_motion.size(-1) in {582, 584}:
            root_start_index = 570
        df_motion[:, :, root_start_index+1] = 0.  # -11 -> 421
        df_motion[:, :, root_start_index+4] = 0.  # -8 -> 424
        df_motion[:, :, root_start_index+6] = 0.  # -6 -> 426
        df_motion[:, :, root_start_index+8] = 0.  # -4 -> 428
        df_motion[:, :, root_start_index+10] = 0.  # -2 -> 430
        df_motion[:, :, root_start_index+7] = 1.  # -5 -> 427

        # Normalize
        norm = torch.sqrt(df_motion[:, :, root_start_index+3] ** 2 + df_motion[:, :, root_start_index+5] ** 2)  # -9 -> 423, -7 -> 425
        df_motion_normalized = df_motion.clone()
        df_motion_normalized[:, :, root_start_index+3] = df_motion[:, :, root_start_index+3] / norm
        df_motion_normalized[:, :, root_start_index+5] = df_motion[:, :, root_start_index+5] / norm

        # Assignment operation
        df_motion[:, :, root_start_index+11] = df_motion[:, :, root_start_index+3]  # -1 -> 431, -9 -> 423
        df_motion[:, :, root_start_index+9] = -df_motion[:, :, root_start_index+5]  # -3 -> 429, -7 -> 425
        return df_motion

    def motion_postprocess_gen(self, df_motion):
        # Ensure the last dimension of motion is 432
        assert df_motion.size(-1) in {434, 432, 417,
                                      419, 582, 584}, "The last dimension of df_motion must be 434, 432, 417, or 419."
        root_start_index = 0
        if df_motion.size(-1) in {432, 434}:
            root_start_index = 420
        elif df_motion.size(-1) in {417, 419}:
            root_start_index = 405
        elif df_motion.size(-1) in {582, 584}:
            root_start_index = 570
        df_motion[:, :, root_start_index+1] = 0.  # -11 -> 421
        df_motion[:, :, root_start_index+4] = 0.  # -8 -> 424
        df_motion[:, :, root_start_index+6] = 0.  # -6 -> 426
        df_motion[:, :, root_start_index+8] = 0.  # -4 -> 428
        df_motion[:, :, root_start_index+10] = 0.  # -2 -> 430
        df_motion[:, :, root_start_index+7] = 1.  # -5 -> 427

        df_normalized = df_motion
        sin_cos = df_normalized[:, :, [root_start_index+3, root_start_index+5]]  # Extract [sin(theta), cos(theta)] or [x, z] components

        norm = torch.norm(sin_cos, p=2, dim=-1, keepdim=True) + 1e-8  # avoid division by zero
        sin_cos_normalized = sin_cos / norm

        df_normalized[:, :, root_start_index+3] = sin_cos_normalized[..., 0]  # update x component
        df_normalized[:, :, root_start_index+5] = sin_cos_normalized[..., 1]  # update z component

        df_normalized[:, :, root_start_index+9] = -sin_cos_normalized[..., 1]  # position corresponding to the -z component
        df_normalized[:, :, root_start_index+11] = sin_cos_normalized[..., 0]  # position corresponding to the x component

        return df_motion

    def generated_phase_postprocess(self, df_phase):
        df_phase = df_phase.permute(0, 2, 1)
        # Ensure the last dimension of motion is 432
        assert df_phase.size(-1) in {527}, "The second dimension of df_phase must be 527."

        root_start_index = 515
        df_phase[:, :, root_start_index+1] = 0.
        df_phase[:, :, root_start_index+4] = 0.
        df_phase[:, :, root_start_index+6] = 0.
        df_phase[:, :, root_start_index+8] = 0.
        df_phase[:, :, root_start_index+10] = 0.
        df_phase[:, :, root_start_index+7] = 1.

        # Normalize
        norm = torch.sqrt(df_phase[:, :, root_start_index+3] ** 2 + df_phase[:, :, root_start_index+5] ** 2)
        df_motion_normalized = df_phase.clone()
        df_motion_normalized[:, :, root_start_index+3] = df_phase[:, :, root_start_index+3] / norm
        df_motion_normalized[:, :, root_start_index+5] = df_phase[:, :, root_start_index+5] / norm

        # Assignment operation
        df_phase[:, :, root_start_index+11] = df_phase[:, :, root_start_index+3]
        df_phase[:, :, root_start_index+9] = -df_phase[:, :, root_start_index+5]
        return df_phase.permute(0, 2, 1)

    def generated_phase_postprocess_gen(self, df_phase):
        df_phase = df_phase.permute(0, 2, 1)
        # Ensure the last dimension of motion is 432
        assert df_phase.size(-1) in {527}, "The second dimension of df_phase must be 527."

        root_start_index = 515
        df_phase[:, :, root_start_index+1] = 0.
        df_phase[:, :, root_start_index+4] = 0.
        df_phase[:, :, root_start_index+6] = 0.
        df_phase[:, :, root_start_index+8] = 0.
        df_phase[:, :, root_start_index+10] = 0.
        df_phase[:, :, root_start_index+7] = 1.

        df_normalized = df_phase
        sin_cos = df_normalized[:, :, [root_start_index+3, root_start_index+5]]  # Extract [sin(theta), cos(theta)] or [x, z] components

        norm = torch.norm(sin_cos, p=2, dim=-1, keepdim=True) + 1e-8  # avoid division by zero
        sin_cos_normalized = sin_cos / norm

        df_normalized[:, :, root_start_index+3] = sin_cos_normalized[..., 0]  # update x component
        df_normalized[:, :, root_start_index+5] = sin_cos_normalized[..., 1]  # update z component

        df_normalized[:, :, root_start_index+9] = -sin_cos_normalized[..., 1]  # position corresponding to the -z component
        df_normalized[:, :, root_start_index+11] = sin_cos_normalized[..., 0]  # position corresponding to the x component
        return df_phase.permute(0, 2, 1)

    def generated_phase_postprocess_manifold(self, df_phase, n_manifold):
        """Postprocess for manifold-based diffusion: normalize trajectory rotation only."""
        df_phase = df_phase.permute(0, 2, 1)
        # Layout: [manifold(n_manifold) | style(1) | traj(12)]
        root_start_index = n_manifold + 1  # skip manifold + style
        if df_phase.size(-1) > root_start_index + 11:
            df_phase[:, :, root_start_index+1] = 0.
            df_phase[:, :, root_start_index+4] = 0.
            df_phase[:, :, root_start_index+6] = 0.
            df_phase[:, :, root_start_index+8] = 0.
            df_phase[:, :, root_start_index+10] = 0.
            df_phase[:, :, root_start_index+7] = 1.

            sin_cos = df_phase[:, :, [root_start_index+3, root_start_index+5]]
            norm = torch.norm(sin_cos, p=2, dim=-1, keepdim=True) + 1e-8
            sin_cos_normalized = sin_cos / norm
            df_phase[:, :, root_start_index+3] = sin_cos_normalized[..., 0]
            df_phase[:, :, root_start_index+5] = sin_cos_normalized[..., 1]
            df_phase[:, :, root_start_index+9] = -sin_cos_normalized[..., 1]
            df_phase[:, :, root_start_index+11] = sin_cos_normalized[..., 0]
        return df_phase.permute(0, 2, 1)

    def compute_text2phase_losses(self, target, model_output, mask):
        # Ensure input shapes match
        batch, feat, frame = target.shape
        assert target.shape == model_output.shape  # [64, 515, 208]
        assert mask.shape == (batch, frame, 1)

        # Split features
        phase_dim = 512
        phase_logits = model_output[:, :phase_dim, :].permute(0, 2, 1)
        angle_output = model_output[:, phase_dim:phase_dim+2, :]
        style_output = model_output[:, phase_dim+2:phase_dim+3, :]
        traj_output = model_output[:, phase_dim+3:, :]

        phase_target = target[:, :phase_dim, :].permute(0, 2, 1)
        angle_target = target[:, phase_dim:phase_dim+2, :]
        style_target = target[:, phase_dim+2:phase_dim+3, :]
        traj_target = target[:, phase_dim+3:, :]

        # Convert mask
        valid_mask = mask.squeeze(-1)  # [64, 208]

        # 1. Phaseclassification loss
        phase_class = torch.argmax(phase_target, dim=-1)
        ce_loss = F.cross_entropy(
            phase_logits.reshape(-1, 512),  # [B*frame, 512]
            phase_class.reshape(-1),  # [B*frame]
            reduction='none'
        ).view(batch, frame)  # [B, frame]
        ce_loss = (ce_loss * valid_mask).sum() / (valid_mask.sum() + 1e-8)

        # 2. angle loss
        angle_mse = F.mse_loss(angle_output, angle_target, reduction='none')
        angle_mse = (angle_mse * mask.permute(0, 2, 1)).sum() / (mask.sum() + 1e-8)


        # 3. Style Codeloss
        style_loss = F.mse_loss(style_output, style_target, reduction='none')
        style_loss = (style_loss * mask.permute(0, 2, 1)).sum() / (mask.sum() + 1e-8)

        traj_mse = F.mse_loss(traj_output, traj_target, reduction='none')
        traj_mse = (traj_mse * mask.permute(0, 2, 1)).sum() / (mask.sum() + 1e-8)

        terms = {
            "phase_ce": ce_loss,
            "angle_mse": angle_mse,
            "style_mse": style_loss,
            "traj_mse": traj_mse,
            "loss": ce_loss * 50.0 + angle_mse * 2.0 + style_loss * 1.0 + traj_mse * 10.0
        }
        return terms

    def compute_text2phase_manifold_losses(self, target, model_output, mask):
        """Compute losses for manifold-based text2phase diffusion (all MSE, no CE)."""
        batch, feat, frame = target.shape
        assert target.shape == model_output.shape
        assert mask.shape == (batch, frame, 1)

        # Layout: [manifold(n_manifold) | style(1) | traj(12)]
        n_manifold = feat - 13  # total - style(1) - traj(12)

        manifold_output = model_output[:, :n_manifold, :]
        style_output = model_output[:, n_manifold:n_manifold+1, :]
        traj_output = model_output[:, n_manifold+1:, :]

        manifold_target = target[:, :n_manifold, :]
        style_target = target[:, n_manifold:n_manifold+1, :]
        traj_target = target[:, n_manifold+1:, :]

        mask_feat = mask.permute(0, 2, 1)  # [B, 1, T]

        # 1. Manifold MSE
        manifold_mse = F.mse_loss(manifold_output, manifold_target, reduction='none')
        manifold_mse = (manifold_mse * mask_feat).sum() / (mask_feat.sum() * n_manifold + 1e-8)

        # 2. Style MSE
        style_mse = F.mse_loss(style_output, style_target, reduction='none')
        style_mse = (style_mse * mask_feat).sum() / (mask_feat.sum() + 1e-8)

        # 3. Trajectory MSE
        traj_mse = F.mse_loss(traj_output, traj_target, reduction='none')
        traj_mse = (traj_mse * mask_feat).sum() / (mask_feat.sum() * 12 + 1e-8)

        terms = {
            "manifold_mse": manifold_mse,
            "style_mse": style_mse,
            "traj_mse": traj_mse,
            "loss": manifold_mse * 1.0 + style_mse * 1.0 + traj_mse * 10.0
        }
        return terms
    def training_text2phase_losses(self, model, x_start, t, model_kwargs=None, noise=None, dataset=None):
        """
        Compute training losses for a single timestep.

        :param model: the model to evaluate loss on.
        :param x_start: the [N x C x ...] tensor of inputs.
        :param t: a batch of timestep indices.
        :param model_kwargs: if not None, a dict of extra keyword arguments to
            pass to the model. This can be used for conditioning.
        :param noise: if specified, the specific Gaussian noise to try to remove.
        :return: a dict with the key "loss" containing a tensor of shape [N].
                 Some mean or variance settings may also have other keys.
        """

        if model_kwargs is None:
            model_kwargs = {}
        if noise is None:
            noise = th.randn_like(x_start)
        x_t = self.q_sample(x_start, t, noise=noise)

        terms = {}

        if self.loss_type == LossType.KL or self.loss_type == LossType.RESCALED_KL:
            terms["loss"] = self._vb_terms_bpd(
                model=model,
                x_start=x_start,
                x_t=x_t,
                t=t,
                clip_denoised=False,
                model_kwargs=model_kwargs,
            )["output"]
            if self.loss_type == LossType.RESCALED_KL:
                terms["loss"] *= self.num_timesteps
        elif self.loss_type == LossType.MSE or self.loss_type == LossType.RESCALED_MSE:
            model_output = model(x_t, self._scale_timesteps(t), **model_kwargs)

            # Check if using manifold mode
            use_manifold = getattr(model, 'use_manifold', False)
            if hasattr(model, 'model'):
                use_manifold = getattr(model.model, 'use_manifold', use_manifold)

            if use_manifold:
                n_manifold = model_output.shape[1] - 13
                model_output = self.generated_phase_postprocess_manifold(model_output, n_manifold)
            else:
                model_output = self.generated_phase_postprocess(model_output)

            target = {
                ModelMeanType.PREVIOUS_X: self.q_posterior_mean_variance(
                    x_start=x_start, x_t=x_t, t=t
                )[0],
                ModelMeanType.START_X: x_start,
                ModelMeanType.EPSILON: noise,
            }[self.model_mean_type]
            assert model_output.shape == target.shape == x_start.shape  # [bs, njoints, nfeats, nframes]
            mask = model_kwargs['y']['mask']

            if use_manifold:
                terms = self.compute_text2phase_manifold_losses(target, model_output, mask)
            else:
                terms = self.compute_text2phase_losses(target, model_output, mask)

        else:
            raise NotImplementedError(self.loss_type)

        return terms
    def training_losses(self, model, x_start, t, model_kwargs=None, noise=None, dataset=None):
        """
        Compute training losses for a single timestep.

        :param model: the model to evaluate loss on.
        :param x_start: the [N x C x ...] tensor of inputs.
        :param t: a batch of timestep indices.
        :param model_kwargs: if not None, a dict of extra keyword arguments to
            pass to the model. This can be used for conditioning.
        :param noise: if specified, the specific Gaussian noise to try to remove.
        :return: a dict with the key "loss" containing a tensor of shape [N].
                 Some mean or variance settings may also have other keys.
        """

        # enc = model.model._modules['module']
        enc = model.model
        # mask = model_kwargs['y']['mask']
        # get_xyz = lambda sample: enc.rot2xyz(sample, mask=None, pose_rep=enc.pose_rep, translation=enc.translation,
        #                                      glob=enc.glob,
        #                                      # jointstype='vertices',  # 3.4 iter/sec # USED ALSO IN MotionCLIP
        #                                      jointstype='smpl',  # 3.4 iter/sec
        #                                      vertstrans=False)

        if model_kwargs is None:
            model_kwargs = {}
        if noise is None:
            noise = th.randn_like(x_start)
        x_t = self.q_sample(x_start, t, noise=noise)

        terms = {}

        if self.loss_type == LossType.KL or self.loss_type == LossType.RESCALED_KL:
            terms["loss"] = self._vb_terms_bpd(
                model=model,
                x_start=x_start,
                x_t=x_t,
                t=t,
                clip_denoised=False,
                model_kwargs=model_kwargs,
            )["output"]
            if self.loss_type == LossType.RESCALED_KL:
                terms["loss"] *= self.num_timesteps
        elif self.loss_type == LossType.MSE or self.loss_type == LossType.RESCALED_MSE:
            motion_cond = model_kwargs['y']['text_embed']
            relative2start_root = model_kwargs['y']['relative2start_rootpos']
            stylecode = model_kwargs['y']['stylecode']
            # motion_cond = torch.zeros_like(motion_cond)
            # mask = torch.zeros_like(motion_cond)
            # mask = (torch.rand_like(motion_cond) > 0.99).float()  # randomly generate 50% zeros and ones
            # Apply mask and set 50% of values to 0
            # motion_cond = motion_cond * mask
            x_t_withcond = torch.cat((x_t, stylecode, motion_cond, relative2start_root), dim=1)
            # if model_kwargs['y']['text_embed'].dim() == 3:
            #     Batch, Phase_dim, Frame = model_kwargs['y']['text_embed'].shape
            #     model_kwargs['y']['text_embed'] = model_kwargs['y']['text_embed'].permute(0, 2, 1).reshape(Batch, Frame * Phase_dim)
            model_output = model(x_t_withcond, self._scale_timesteps(t), **model_kwargs)
            model_output = self.motion_postprocess_train(model_output.permute(0, 2, 1)).permute(0, 2, 1)
            split_sizes = [model_output.size(1) - 2, 2]
            model_output, df_contact = torch.split(model_output, split_sizes, dim=1)
            # if self.model_var_type in [
            #     ModelVarType.LEARNED,
            #     ModelVarType.LEARNED_RANGE,
            # ]:
            #     B, C = x_t.shape[:2]
            #     assert model_output.shape == (B, C * 2, *x_t.shape[2:])
            #     model_output, model_var_values = th.split(model_output, C, dim=1)
            #     # Learn the variance using the variational bound, but don't let
            #     # it affect our mean prediction.
            #     frozen_out = th.cat([model_output.detach(), model_var_values], dim=1)
            #     terms["vb"] = self._vb_terms_bpd(
            #         model=lambda *args, r=frozen_out: r,
            #         x_start=x_start,
            #         x_t=x_t,
            #         t=t,
            #         clip_denoised=False,
            #     )["output"]
            #     if self.loss_type == LossType.RESCALED_MSE:
            #         # Divide by 1000 for equivalence with initial implementation.
            #         # Without a factor of 1/1000, the VB term hurts the MSE term.
            #         terms["vb"] *= self.num_timesteps / 1000.0

            target = {
                ModelMeanType.PREVIOUS_X: self.q_posterior_mean_variance(
                    x_start=x_start, x_t=x_t, t=t
                )[0],
                ModelMeanType.START_X: x_start,
                ModelMeanType.EPSILON: noise,
            }[self.model_mean_type]
            # target_debug = target.cpu().numpy()
            # model_output_debug = model_output.detach().cpu().numpy()
            assert model_output.shape == target.shape == x_start.shape  # [bs, njoints, nfeats, nframes]
            # terms["rot_mse"] = self.masked_l2(target, model_output, mask) # mean_flat(rot_mse)
            mse_loss = F.mse_loss(target, model_output, reduction="none")
            # global_pos = self.transfer2absolute_batch(model_output.permute(0, 2, 1))[:, :, -12:-9]
            # golbal_pos_gt = model_kwargs['y']['relative2start_rootpos']
            # terms["global_pos_mse"] = F.mse_loss(global_pos, golbal_pos_gt, reduction="none").mean(dim=[1, 2])

            main_weight = 100.0  # weight of the main loss
            tail_weight = 50000.0  # weight of the last 12 dimensions

            main_loss = mse_loss[..., :-12, :] * main_weight  # weight the preceding dimensions
            tail_loss = mse_loss[..., -12:, :] * tail_weight  # weight the last 12 dimensions

            terms["mse"] = torch.cat([main_loss, tail_loss], dim=1).mean(dim=[1, 2])


            # foot contact loss
            foot_contact = model_kwargs['y']['foot_contact']
            # motion_debug = model_output.permute(0, 2, 1).detach().cpu().numpy()

            # debug
            # global_pos_gt = self.transfer2absolute_batch(target.permute(0, 2, 1))
            # global_pos_gt = global_pos_gt.reshape(-1, global_pos_gt.shape[2]).detach().cpu().numpy()
            # frame_num = global_pos_gt.shape[0]
            # global_pos_gt = global_pos_gt * dataset.data_std + dataset.data_mean
            # write_motion2npz(global_pos_gt, dataset.data_std, dataset.data_mean, frame_num,
            #                  f"results/visualnpz/output_motion{7}.npz", True)

            # global_pos[:, :, -9:] = global_pos_gt[:, :, -9:]
            data_std_tensor = torch.tensor(dataset.data_std, device='cuda', dtype=torch.float32)
            data_mean_tensor = torch.tensor(dataset.data_mean, device='cuda', dtype=torch.float32)

            # debug
            # global_pos_gt = global_pos.reshape(-1, global_pos.shape[2]).detach().cpu().numpy()
            # frame_num = global_pos_gt.shape[0]
            # write_motion2npz(global_pos_gt, dataset.data_std, dataset.data_mean, frame_num,
            #                  f"results/visualnpz/output_motion{0}.npz", True)
            assert model_output.size(-2) in {432, 417, 582}, "The feature dimension of model_output must be 432 or 417."
            global_pos_1 = model_output.permute(0, 2, 1) * data_std_tensor + data_mean_tensor
            # the pattern (Position) is right toe, left toe, right toe site, left toe site
            if model_output.size(-2) == 432:
                foot_indices = [147, 148, 149, 162, 163, 164, 150, 151, 152, 165, 166, 167]
            elif model_output.size(-2) == 417:
                foot_indices = [108, 109, 110, 93, 94, 95, 111, 112, 113, 96, 97, 98]
            elif model_output.size(-2) == 582:
                foot_indices = [147, 148, 149, 129, 130, 131, 150, 151, 152, 132, 133, 134]

            foot_pos = global_pos_1[..., :, foot_indices]  # shape: [batch, frames, 12]
            foot_pos = foot_pos.reshape(foot_pos.shape[0], foot_pos.shape[1], 4, 3)

            foot_pos_laterf = foot_pos[:, 1:, ...]
            foot_pos_formerf = foot_pos[:, :-1, ...]

            root_info_pos = global_pos_1[:, 1:, -12:-9]
            root_info_rot = global_pos_1[:, 1:, -9:].reshape(root_info_pos.shape[0], root_info_pos.shape[1], 3, 3)
            foot_pos_later2former = torch.matmul(foot_pos_laterf,
                                                 root_info_rot.transpose(-1, -2)) + root_info_pos.unsqueeze(-2)

            foot_pos_loss = torch.norm(foot_pos_later2former-foot_pos_formerf, dim=-1)
            foot_pos_loss = (foot_pos_loss * foot_contact[:, 1:, :].repeat(1, 1, 2)).mean(dim=[1, 2])
            terms["foot_contact_loss"] = foot_pos_loss
            foot_contact_loss_weight = self.lambda_fc

            # contact label loss
            contact_bce_loss = F.binary_cross_entropy_with_logits(df_contact, foot_contact.permute(0, 2, 1),
                                                                  reduction="none")
            terms["contact_label_loss"] = contact_bce_loss.mean(dim=[1, 2])
            label_loss_w = 1.
            terms["loss"] = terms["mse"] + foot_contact_loss_weight * terms["foot_contact_loss"] + label_loss_w * terms["contact_label_loss"]

            # global_root_loss
            # global_root_gt = model_kwargs['y']['global_root'].permute(0, 2, 1)
            # global_pos = self.transfer2absolute_batch(model_output.permute(0, 2, 1))
            # global_root_df = global_pos[..., -12:]
            # terms["global_root_mse"] = F.mse_loss(global_root_gt, global_root_df, reduction="none").mean(dim=[1, 2])
            # terms["loss"] = terms["mse"] + 0.1 * terms["global_root_mse"]
            # terms["loss"] = terms["mse"]
        else:
            raise NotImplementedError(self.loss_type)

        return terms
    def single_shot_noise_generation(self, model, t, gt_motion, model_kwargs=None, noise=None, dataset=None):
        """
        Compute training losses for a single timestep.

        :param model: the model to evaluate loss on.
        :param x_start: the [N x C x ...] tensor of inputs.
        :param t: a batch of timestep indices.
        :param model_kwargs: if not None, a dict of extra keyword arguments to
            pass to the model. This can be used for conditioning.
        :param noise: if specified, the specific Gaussian noise to try to remove.
        :return: a dict with the key "loss" containing a tensor of shape [N].
                 Some mean or variance settings may also have other keys.
        """
        if model_kwargs is None:
            model_kwargs = {}
        if noise is None:
            noise = th.randn_like(gt_motion)
        init_image = th.zeros_like(gt_motion)
        x_t = self.q_sample(init_image, t, noise=noise)

        terms = {}

        motion_cond = model_kwargs['y']['text_embed']
        x_t_withcond = torch.cat((x_t, motion_cond), dim=1)
        model_output = model(x_t_withcond, self._scale_timesteps(t))
        target = {
            ModelMeanType.PREVIOUS_X: self.q_posterior_mean_variance(
                x_start=init_image, x_t=x_t, t=t
            )[0],
            ModelMeanType.START_X: init_image,
            ModelMeanType.EPSILON: noise,
        }[self.model_mean_type]
        assert model_output.shape == target.shape == init_image.shape  # [bs, njoints, nfeats, nframes]
        # terms["rot_mse"] = self.masked_l2(target, model_output, mask) # mean_flat(rot_mse)
        terms["mse"] = F.mse_loss(gt_motion, model_output)
        # if t[0] == 0:
        #     frame_num = 61 * 10
        #     motion = gt_motion.permute(0, 2, 1).cpu()
        #     motion = motion.reshape(-1, motion.shape[-1])
        #     write_motion2npz(motion, dataset.data_std, dataset.data_mean, frame_num,
        #                      './results/difftest5/generate/gt_motion_train.npz')
        #     motion = model_output.permute(0, 2, 1).detach().cpu()
        #     motion = motion.reshape(-1, motion.shape[-1])
        #     write_motion2npz(motion, dataset.data_std, dataset.data_mean, frame_num,
        #                      './results/difftest5/generate/df_motion_train.npz')
        terms["loss"] = terms["mse"]

        return terms, model_output

    def single_progressive_training_generation(self, model, x_start, t, gt_motion, model_kwargs=None, noise=None, dataset=None):
        """
        Compute training losses for a single timestep.

        :param model: the model to evaluate loss on.
        :param x_start: the [N x C x ...] tensor of inputs.
        :param t: a batch of timestep indices.
        :param model_kwargs: if not None, a dict of extra keyword arguments to
            pass to the model. This can be used for conditioning.
        :param noise: if specified, the specific Gaussian noise to try to remove.
        :return: a dict with the key "loss" containing a tensor of shape [N].
                 Some mean or variance settings may also have other keys.
        """
        if model_kwargs is None:
            model_kwargs = {}
        if noise is None:
            noise = th.randn_like(x_start)
        x_t = self.q_sample(x_start, t, noise=noise)

        terms = {}

        if self.loss_type == LossType.KL or self.loss_type == LossType.RESCALED_KL:
            terms["loss"] = self._vb_terms_bpd(
                model=model,
                x_start=x_start,
                x_t=x_t,
                t=t,
                clip_denoised=False,
                model_kwargs=model_kwargs,
            )["output"]
            if self.loss_type == LossType.RESCALED_KL:
                terms["loss"] *= self.num_timesteps
        elif self.loss_type == LossType.MSE or self.loss_type == LossType.RESCALED_MSE:
            motion_cond = model_kwargs['y']['text_embed']
            x_t_withcond = torch.cat((x_t, motion_cond), dim=1)
            model_output = model(x_t_withcond, self._scale_timesteps(t))

            target = {
                ModelMeanType.PREVIOUS_X: self.q_posterior_mean_variance(
                    x_start=x_start, x_t=x_t, t=t
                )[0],
                ModelMeanType.START_X: x_start,
                ModelMeanType.EPSILON: noise,
            }[self.model_mean_type]
            assert model_output.shape == target.shape == x_start.shape  # [bs, njoints, nfeats, nframes]

            # terms["rot_mse"] = self.masked_l2(target, model_output, mask) # mean_flat(rot_mse)
            terms["mse"] = F.mse_loss(gt_motion, model_output)

            # if t[0] == 0:
            #     frame_num = 61 * 10
            #     motion = gt_motion.permute(0, 2, 1).cpu()
            #     motion = motion.reshape(-1, motion.shape[-1])
            #     write_motion2npz(motion, dataset.data_std, dataset.data_mean, frame_num,
            #                      './results/difftest5/generate/gt_motion_train.npz')
            #     motion = model_output.permute(0, 2, 1).detach().cpu()
            #     motion = motion.reshape(-1, motion.shape[-1])
            #     write_motion2npz(motion, dataset.data_std, dataset.data_mean, frame_num,
            #                      './results/difftest5/generate/df_motion_train.npz')

            terms["loss"] = terms["mse"]
        else:
            raise NotImplementedError(self.loss_type)

        return terms, model_output

    def fc_loss_rot_repr(self, gt_xyz, pred_xyz, mask):
        def to_np_cpu(x):
            return x.detach().cpu().numpy()
        """
        pose_xyz: SMPL batch tensor of shape: [BatchSize, 24, 3, Frames]
        """
        # 'L_Ankle',  # 7, 'R_Ankle',  # 8 , 'L_Foot',  # 10, 'R_Foot',  # 11

        l_ankle_idx, r_ankle_idx = 7, 8
        l_foot_idx, r_foot_idx = 10, 11
        """ Contact calculated by 'Kfir Method' Commented code)"""
        # contact_signal = torch.zeros((pose_xyz.shape[0], pose_xyz.shape[3], 2), device=pose_xyz.device) # [BatchSize, Frames, 2]
        # left_xyz = 0.5 * (pose_xyz[:, l_ankle_idx, :, :] + pose_xyz[:, l_foot_idx, :, :]) # [BatchSize, 3, Frames]
        # right_xyz = 0.5 * (pose_xyz[:, r_ankle_idx, :, :] + pose_xyz[:, r_foot_idx, :, :])
        # left_z, right_z = left_xyz[:, 2, :], right_xyz[:, 2, :] # [BatchSize, Frames]
        # left_velocity = torch.linalg.norm(left_xyz[:, :, 2:] - left_xyz[:, :, :-2], axis=1)  # [BatchSize, Frames]
        # right_velocity = torch.linalg.norm(left_xyz[:, :, 2:] - left_xyz[:, :, :-2], axis=1)
        #
        # left_z_mask = left_z <= torch.mean(torch.sort(left_z)[0][:, :left_z.shape[1] // 5], axis=-1)
        # left_z_mask = torch.stack([left_z_mask, left_z_mask], dim=-1) # [BatchSize, Frames, 2]
        # left_z_mask[:, :, 1] = False  # Blank right side
        # contact_signal[left_z_mask] = 0.4
        #
        # right_z_mask = right_z <= torch.mean(torch.sort(right_z)[0][:, :right_z.shape[1] // 5], axis=-1)
        # right_z_mask = torch.stack([right_z_mask, right_z_mask], dim=-1) # [BatchSize, Frames, 2]
        # right_z_mask[:, :, 0] = False  # Blank left side
        # contact_signal[right_z_mask] = 0.4
        # contact_signal[left_z <= (torch.mean(torch.sort(left_z)[:left_z.shape[0] // 5]) + 20), 0] = 1
        # contact_signal[right_z <= (torch.mean(torch.sort(right_z)[:right_z.shape[0] // 5]) + 20), 1] = 1

        # plt.plot(to_np_cpu(left_z[0]), label='left_z')
        # plt.plot(to_np_cpu(left_velocity[0]), label='left_velocity')
        # plt.plot(to_np_cpu(contact_signal[0, :, 0]), label='left_fc')
        # plt.grid()
        # plt.legend()
        # plt.show()
        # plt.plot(to_np_cpu(right_z[0]), label='right_z')
        # plt.plot(to_np_cpu(right_velocity[0]), label='right_velocity')
        # plt.plot(to_np_cpu(contact_signal[0, :, 1]), label='right_fc')
        # plt.grid()
        # plt.legend()
        # plt.show()

        gt_joint_xyz = gt_xyz[:, [l_ankle_idx, l_foot_idx, r_ankle_idx, r_foot_idx], :, :]  # [BatchSize, 4, 3, Frames]
        gt_joint_vel = torch.linalg.norm(gt_joint_xyz[:, :, :, 1:] - gt_joint_xyz[:, :, :, :-1], axis=2)  # [BatchSize, 4, Frames]
        fc_mask = (gt_joint_vel <= 0.01)
        pred_joint_xyz = pred_xyz[:, [l_ankle_idx, l_foot_idx, r_ankle_idx, r_foot_idx], :, :]  # [BatchSize, 4, 3, Frames]
        pred_joint_vel = torch.linalg.norm(pred_joint_xyz[:, :, :, 1:] - pred_joint_xyz[:, :, :, :-1], axis=2)  # [BatchSize, 4, Frames]
        pred_joint_vel[~fc_mask] = 0  # Blank non-contact velocities frames. [BS,4,FRAMES]
        pred_joint_vel = torch.unsqueeze(pred_joint_vel, dim=2)

        """DEBUG CODE"""
        # print(f'mask: {mask.shape}')
        # print(f'pred_joint_vel: {pred_joint_vel.shape}')
        # plt.title(f'Joint: {joint_idx}')
        # plt.plot(to_np_cpu(gt_joint_vel[0]), label='velocity')
        # plt.plot(to_np_cpu(fc_mask[0]), label='fc')
        # plt.grid()
        # plt.legend()
        # plt.show()
        return self.masked_l2(pred_joint_vel, torch.zeros(pred_joint_vel.shape, device=pred_joint_vel.device),
                              mask[:, :, :, 1:])
    # TODO - NOT USED YET, JUST COMMITING TO NOT DELETE THIS AND KEEP INITIAL IMPLEMENTATION, NOT DONE!
    def foot_contact_loss_humanml3d(self, target, model_output):
        # root_rot_velocity (B, seq_len, 1)
        # root_linear_velocity (B, seq_len, 2)
        # root_y (B, seq_len, 1)
        # ric_data (B, seq_len, (joint_num - 1)*3) , XYZ
        # rot_data (B, seq_len, (joint_num - 1)*6) , 6D
        # local_velocity (B, seq_len, joint_num*3) , XYZ
        # foot contact (B, seq_len, 4) ,

        target_fc = target[:, -4:, :, :]
        root_rot_velocity = target[:, :1, :, :]
        root_linear_velocity = target[:, 1:3, :, :]
        root_y = target[:, 3:4, :, :]
        ric_data = target[:, 4:67, :, :]  # 4+(3*21)=67
        rot_data = target[:, 67:193, :, :]  # 67+(6*21)=193
        local_velocity = target[:, 193:259, :, :]  # 193+(3*22)=259
        contact = target[:, 259:, :, :]  # 193+(3*22)=259
        contact_mask_gt = contact > 0.5  # contact mask order for indexes are fid_l [7, 10], fid_r [8, 11]
        vel_lf_7 = local_velocity[:, 7 * 3:8 * 3, :, :]
        vel_rf_8 = local_velocity[:, 8 * 3:9 * 3, :, :]
        vel_lf_10 = local_velocity[:, 10 * 3:11 * 3, :, :]
        vel_rf_11 = local_velocity[:, 11 * 3:12 * 3, :, :]

        calc_vel_lf_7 = ric_data[:, 6 * 3:7 * 3, :, 1:] - ric_data[:, 6 * 3:7 * 3, :, :-1]
        calc_vel_rf_8 = ric_data[:, 7 * 3:8 * 3, :, 1:] - ric_data[:, 7 * 3:8 * 3, :, :-1]
        calc_vel_lf_10 = ric_data[:, 9 * 3:10 * 3, :, 1:] - ric_data[:, 9 * 3:10 * 3, :, :-1]
        calc_vel_rf_11 = ric_data[:, 10 * 3:11 * 3, :, 1:] - ric_data[:, 10 * 3:11 * 3, :, :-1]

        # vel_foots = torch.stack([vel_lf_7, vel_lf_10, vel_rf_8, vel_rf_11], dim=1)
        for chosen_vel_foot_calc, chosen_vel_foot, joint_idx, contact_mask_idx in zip(
                [calc_vel_lf_7, calc_vel_rf_8, calc_vel_lf_10, calc_vel_rf_11],
                [vel_lf_7, vel_lf_10, vel_rf_8, vel_rf_11],
                [7, 10, 8, 11],
                [0, 1, 2, 3]):
            tmp_mask_gt = contact_mask_gt[:, contact_mask_idx, :, :].cpu().detach().numpy().reshape(-1).astype(int)
            chosen_vel_norm = np.linalg.norm(chosen_vel_foot.cpu().detach().numpy().reshape((3, -1)), axis=0)
            chosen_vel_calc_norm = np.linalg.norm(chosen_vel_foot_calc.cpu().detach().numpy().reshape((3, -1)),
                                                  axis=0)

            print(tmp_mask_gt.shape)
            print(chosen_vel_foot.shape)
            print(chosen_vel_calc_norm.shape)
            import matplotlib.pyplot as plt
            plt.plot(tmp_mask_gt, label='FC mask')
            plt.plot(chosen_vel_norm, label='Vel. XYZ norm (from vector)')
            plt.plot(chosen_vel_calc_norm, label='Vel. XYZ norm (calculated diff XYZ)')

            plt.title(f'FC idx {contact_mask_idx}, Joint Index {joint_idx}')
            plt.legend()
            plt.show()
        # print(vel_foots.shape)
        return 0
    # TODO - NOT USED YET, JUST COMMITING TO NOT DELETE THIS AND KEEP INITIAL IMPLEMENTATION, NOT DONE!
    def velocity_consistency_loss_humanml3d(self, target, model_output):
        # root_rot_velocity (B, seq_len, 1)
        # root_linear_velocity (B, seq_len, 2)
        # root_y (B, seq_len, 1)
        # ric_data (B, seq_len, (joint_num - 1)*3) , XYZ
        # rot_data (B, seq_len, (joint_num - 1)*6) , 6D
        # local_velocity (B, seq_len, joint_num*3) , XYZ
        # foot contact (B, seq_len, 4) ,

        target_fc = target[:, -4:, :, :]
        root_rot_velocity = target[:, :1, :, :]
        root_linear_velocity = target[:, 1:3, :, :]
        root_y = target[:, 3:4, :, :]
        ric_data = target[:, 4:67, :, :]  # 4+(3*21)=67
        rot_data = target[:, 67:193, :, :]  # 67+(6*21)=193
        local_velocity = target[:, 193:259, :, :]  # 193+(3*22)=259
        contact = target[:, 259:, :, :]  # 193+(3*22)=259

        calc_vel_from_xyz = ric_data[:, :, :, 1:] - ric_data[:, :, :, :-1]
        velocity_from_vector = local_velocity[:, 3:, :, 1:]  # Slicing out root
        r_rot_quat, r_pos = motion_process.recover_root_rot_pos(target.permute(0, 2, 3, 1).type(th.FloatTensor))
        print(f'r_rot_quat: {r_rot_quat.shape}')
        print(f'calc_vel_from_xyz: {calc_vel_from_xyz.shape}')
        calc_vel_from_xyz = calc_vel_from_xyz.permute(0, 2, 3, 1)
        calc_vel_from_xyz = calc_vel_from_xyz.reshape((1, 1, -1, 21, 3)).type(th.FloatTensor)
        r_rot_quat_adapted = r_rot_quat[..., :-1, None, :].repeat((1,1,1,21,1)).to(calc_vel_from_xyz.device)
        print(f'calc_vel_from_xyz: {calc_vel_from_xyz.shape} , {calc_vel_from_xyz.device}')
        print(f'r_rot_quat_adapted: {r_rot_quat_adapted.shape}, {r_rot_quat_adapted.device}')

        calc_vel_from_xyz = motion_process.qrot(r_rot_quat_adapted, calc_vel_from_xyz)
        calc_vel_from_xyz = calc_vel_from_xyz.reshape((1, 1, -1, 21 * 3))
        calc_vel_from_xyz = calc_vel_from_xyz.permute(0, 3, 1, 2)
        print(f'calc_vel_from_xyz: {calc_vel_from_xyz.shape} , {calc_vel_from_xyz.device}')

        import matplotlib.pyplot as plt
        for i in range(21):
            plt.plot(np.linalg.norm(calc_vel_from_xyz[:,i*3:(i+1)*3,:,:].cpu().detach().numpy().reshape((3, -1)), axis=0), label='Calc Vel')
            plt.plot(np.linalg.norm(velocity_from_vector[:,i*3:(i+1)*3,:,:].cpu().detach().numpy().reshape((3, -1)), axis=0), label='Vector Vel')
            plt.title(f'Joint idx: {i}')
            plt.legend()
            plt.show()
        print(calc_vel_from_xyz.shape)
        print(velocity_from_vector.shape)
        diff = calc_vel_from_xyz-velocity_from_vector
        print(np.linalg.norm(diff.cpu().detach().numpy().reshape((63, -1)), axis=0))

        return 0


    def _prior_bpd(self, x_start):
        """
        Get the prior KL term for the variational lower-bound, measured in
        bits-per-dim.

        This term can't be optimized, as it only depends on the encoder.

        :param x_start: the [N x C x ...] tensor of inputs.
        :return: a batch of [N] KL values (in bits), one per batch element.
        """
        batch_size = x_start.shape[0]
        t = th.tensor([self.num_timesteps - 1] * batch_size, device=x_start.device)
        qt_mean, _, qt_log_variance = self.q_mean_variance(x_start, t)
        kl_prior = normal_kl(
            mean1=qt_mean, logvar1=qt_log_variance, mean2=0.0, logvar2=0.0
        )
        return mean_flat(kl_prior) / np.log(2.0)

    def calc_bpd_loop(self, model, x_start, clip_denoised=True, model_kwargs=None):
        """
        Compute the entire variational lower-bound, measured in bits-per-dim,
        as well as other related quantities.

        :param model: the model to evaluate loss on.
        :param x_start: the [N x C x ...] tensor of inputs.
        :param clip_denoised: if True, clip denoised samples.
        :param model_kwargs: if not None, a dict of extra keyword arguments to
            pass to the model. This can be used for conditioning.

        :return: a dict containing the following keys:
                 - total_bpd: the total variational lower-bound, per batch element.
                 - prior_bpd: the prior term in the lower-bound.
                 - vb: an [N x T] tensor of terms in the lower-bound.
                 - xstart_mse: an [N x T] tensor of x_0 MSEs for each timestep.
                 - mse: an [N x T] tensor of epsilon MSEs for each timestep.
        """
        device = x_start.device
        batch_size = x_start.shape[0]

        vb = []
        xstart_mse = []
        mse = []
        for t in list(range(self.num_timesteps))[::-1]:
            t_batch = th.tensor([t] * batch_size, device=device)
            noise = th.randn_like(x_start)
            x_t = self.q_sample(x_start=x_start, t=t_batch, noise=noise)
            # Calculate VLB term at the current timestep
            with th.no_grad():
                out = self._vb_terms_bpd(
                    model,
                    x_start=x_start,
                    x_t=x_t,
                    t=t_batch,
                    clip_denoised=clip_denoised,
                    model_kwargs=model_kwargs,
                )
            vb.append(out["output"])
            xstart_mse.append(mean_flat((out["pred_xstart"] - x_start) ** 2))
            eps = self._predict_eps_from_xstart(x_t, t_batch, out["pred_xstart"])
            mse.append(mean_flat((eps - noise) ** 2))

        vb = th.stack(vb, dim=1)
        xstart_mse = th.stack(xstart_mse, dim=1)
        mse = th.stack(mse, dim=1)

        prior_bpd = self._prior_bpd(x_start)
        total_bpd = vb.sum(dim=1) + prior_bpd
        return {
            "total_bpd": total_bpd,
            "prior_bpd": prior_bpd,
            "vb": vb,
            "xstart_mse": xstart_mse,
            "mse": mse,
        }


def _extract_into_tensor(arr, timesteps, broadcast_shape):
    """
    Extract values from a 1-D numpy array for a batch of indices.

    :param arr: the 1-D numpy array.
    :param timesteps: a tensor of indices into the array to extract.
    :param broadcast_shape: a larger shape of K dimensions with the batch
                            dimension equal to the length of timesteps.
    :return: a tensor of shape [batch_size, 1, ...] where the shape has K dims.
    """
    res = th.from_numpy(arr).to(device=timesteps.device)[timesteps].float()
    while len(res.shape) < len(broadcast_shape):
        res = res[..., None]
    return res.expand(broadcast_shape)
