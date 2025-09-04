from typing import *
import copy
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
from easydict import EasyDict as edict

from ..basic import BasicTrainer
from ...pipelines import samplers
from ...utils.general_utils import dict_reduce
from ...utils.data_utils import set_local_seed
from .mixins.classifier_free_guidance import ClassifierFreeGuidanceMixin
from .mixins.text_conditioned import TextConditionedMixin
from .mixins.image_conditioned import ImageConditionedMixin


class FlowMDTTrainer(BasicTrainer):
    """
    Trainer for diffusion model with MDT-dist objective.
    
    Args:
        models (dict[str, nn.Module]): Models to train.
        dataset (torch.utils.data.Dataset): Dataset.
        output_dir (str): Output directory.
        load_dir (str): Load directory.
        step (int): Step to load.
        batch_size (int): Batch size.
        batch_size_per_gpu (int): Batch size per GPU. If specified, batch_size will be ignored.
        batch_split (int): Split batch with gradient accumulation.
        max_steps (int): Max steps.
        optimizer (dict): Optimizer config.
        lr_scheduler (dict): Learning rate scheduler config.
        elastic (dict): Elastic memory management config.
        grad_clip (float or dict): Gradient clip config.
        ema_rate (float or list): Exponential moving average rates.
        fp16_mode (str): FP16 mode.
            - None: No FP16.
            - 'inflat_all': Hold a inflated fp32 master param for all params.
            - 'amp': Automatic mixed precision.
        fp16_scale_growth (float): Scale growth for FP16 gradient backpropagation.
        finetune_ckpt (dict): Finetune checkpoint.
        log_param_stats (bool): Log parameter stats.
        i_print (int): Print interval.
        i_log (int): Log interval.
        i_sample (int): Sample interval.
        i_save (int): Save interval.
        i_ddpcheck (int): DDP check interval.

        t_schedule (dict): Time schedule for flow matching.
        sigma_min (float): Minimum noise level.
    """
    def __init__(
        self,
        *args,
        t_schedule: dict = {
            'name': 'logitNormal',
            'args': {
                'mean': 1.0,
                'std': 1.0,
            }
        },
        sigma_min: float = 1e-5,
        distill_type: Literal['discrete', 'continuous', 'cm'] = 'discrete',
        distill_interval: Union[list[float], float] = 0.01,
        cfg_teacher_vm: Union[list[float], float] = 40.0,
        cfg_teacher_vd: Union[list[float], float] = 100.0,
        rescale_t: float = 1.0,
        multiphase: int = 1,
        lambda_vm: float = 1.0,
        lambda_vd: float = 1.0,
        d_norm: bool = True,
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.t_schedule = t_schedule
        self.sigma_min = sigma_min # TODO: add sigma_min to diffuse process
        self.distill_type = distill_type
        self.interval = distill_interval
        self.cfg_teacher_vm = cfg_teacher_vm
        self.cfg_teacher_vd = cfg_teacher_vd
        self.rescale_t = rescale_t
        self.multiphase = multiphase
        self.lambda_vm = lambda_vm
        self.lambda_vd = lambda_vd
        self.d_norm = d_norm
        
    def diffuse(self, x_0: torch.Tensor, t: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        """
        Diffuse the data for a given number of diffusion steps.
        In other words, sample from q(x_t | x_0).

        Args:
            x_0: The [N x C x ...] tensor of noiseless inputs.
            t: The [N] tensor of diffusion steps [0-1].
            noise: If specified, use this noise instead of generating new noise.

        Returns:
            x_t, the noisy version of x_0 under timestep t.
        """
        if noise is None:
            noise = torch.randn_like(x_0)
        assert noise.shape == x_0.shape, "noise must have same shape as x_0"
        t = t.view(-1, *[1 for _ in range(len(x_0.shape) - 1)])
        x_t = (1 - t) * x_0 + (self.sigma_min + (1 - self.sigma_min) * t) * noise
        return x_t

    def get_v(self, x_0: torch.Tensor, noise: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Compute the velocity of the diffusion process at time t.
        """
        return (1 - self.sigma_min) * noise - x_0

    def euler_step(self, x_t, v, delta_t):
        """
        Perform a single Euler step.
        """
        delta_t = delta_t.view(-1, *[1 for _ in range(len(x_t.shape) - 1)])
        return x_t - delta_t * v
        # return (1 - self.sigma_min) * x_t - (self.sigma_min + (1 - self.sigma_min) * delta_t) * dxt_dt

    def get_cond(self, cond, **kwargs):
        """
        Get the conditioning data.
        """
        return cond
    
    def get_inference_cond(self, cond, **kwargs):
        """
        Get the conditioning data for inference.
        """
        return {'cond': cond, **kwargs}

    def get_sampler(self, **kwargs) -> samplers.FlowEulerCfgSampler:
        """
        Get the sampler for the diffusion process.
        """
        return samplers.FlowEulerCfgSampler(self.sigma_min)
    
    def vis_cond(self, **kwargs):
        """
        Visualize the conditioning data.
        """
        return {}
    
    def get_target_t(self, t: torch.Tensor) -> torch.Tensor:
        """
        Get the target timestep for multi-phased distillation.
        """
        if self.distill_type in ['discrete', 'cm']:
            t = t - self.interval
        multiphase = self.multiphase
        t_targets = np.linspace(0., 1., num=multiphase, endpoint=False)
        t_targets = torch.from_numpy(t_targets).float()
        t_targets = t_targets.to(t.device)[None].repeat(t.shape[0], 1)
        t = t.view(-1, 1)
        if multiphase > 1:
            idx_targets = (t < t_targets).float().argmax(dim=1) - 1
        else:
            idx_targets = torch.zeros_like(t[:, 0], dtype=torch.long)
        t_target = t_targets[torch.arange(t_targets.shape[0], device=t.device), idx_targets]
        return t_target

    def sample_t(self, batch_size: int, t_schedule: dict = None) -> torch.Tensor:
        """
        Sample timesteps.
        """
        if t_schedule is None:
            t_schedule = self.t_schedule
        if t_schedule['name'] == 'uniform':
            t = torch.rand(batch_size)
        elif t_schedule['name'] == 'logitNormal':
            mean = t_schedule['args']['mean']
            std = t_schedule['args']['std']
            t = torch.sigmoid(torch.randn(batch_size) * std + mean)
        else:
            raise ValueError(f"Unknown t_schedule: {t_schedule['name']}")
        
        if self.distill_type in ['discrete', 'cm']:
            t = self.sigma_min + self.interval + (1. - (self.interval + self.sigma_min)) * t
        elif self.distill_type == 'continuous':
            t = self.sigma_min + (1. - self.sigma_min) * t
        else:
            raise ValueError(f"Unknown distill_type: {self.distill_type}")
        return t
    
    def get_v_pred_tgt(
        self,
        x_0: torch.Tensor,
        noise: torch.Tensor,
        t: torch.Tensor,
        cond: torch.Tensor,
        cfg_teacher: Optional[Union[List[float], float]] = None,
        **kwargs):
        '''
        Compute the prediction and target.
        '''
        cond_train = self.get_cond(cond, **kwargs)
        t_reshape = t.view(-1, *[1 for _ in range(len(x_0.shape) - 1)])
        x_t = self.diffuse(x_0, t, noise)

        # teacher
        if self.teacher_model is not None:
            with torch.no_grad():
                cond_inference_dict = self.get_inference_cond(cond, **kwargs)
                v_teacher = self.teacher_model(x_t, t * 1000, cond_inference_dict['cond'], **kwargs)
                # cfg sample
                if cfg_teacher is not None:
                    if isinstance(cfg_teacher, list):
                        cfg_t = np.random.choice(cfg_teacher)
                    else:
                        cfg_t = cfg_teacher
                    if cfg_t > 0:
                        v_teacher_neg = self.teacher_model(x_t, t * 1000, cond_inference_dict['neg_cond'], **kwargs)
                        v_teacher = (1. + cfg_t) * v_teacher - cfg_t * v_teacher_neg
        else:
            v_teacher = self.get_v(x_0, noise, t)

        # student pred
        v_pred = self.training_models['denoiser'](x_t, t * 1000, cond_train, **kwargs)
        # target
        if self.distill_type == 'continuous':
            with torch.no_grad():
                def model_wrapper(x_t, t):
                    return self.training_models['denoiser'](x_t, t * 1000, cond=cond_train, **kwargs)
                # torch.autograd.functional supports reverse mode, while torch.func only supports forward mode
                _, dv_dt = torch.autograd.functional.jvp(
                    model_wrapper, (x_t, t), (v_teacher, torch.ones_like(t))
                )
            # normalization
            if self.d_norm:
                dv_dt_norm = torch.linalg.vector_norm(dv_dt, dim=tuple(i+1 for i in range(len(x_0.shape) - 1)), keepdim=True)
                dv_dt = dv_dt / (dv_dt_norm + 0.1)  # 0.1 is the constant c, can be modified
            v_tgt = v_teacher - t_reshape * dv_dt
            return v_pred, v_tgt
        elif self.distill_type == 'discrete':
            with torch.no_grad():
                t_prev = t - self.interval
                x_prev = self.euler_step(x_t, v_teacher, t - t_prev)
                v_prev = self.training_models['denoiser'](x_prev, t_prev * 1000, cond_train, **kwargs)
                dv_dt = (v_pred.detach() - v_prev) / self.interval
            # normalization
            if self.d_norm:
                dv_dt_norm = torch.linalg.vector_norm(dv_dt, dim=tuple(i+1 for i in range(len(x_0.shape) - 1)), keepdim=True)
                dv_dt = dv_dt / (dv_dt_norm + 0.1)  # 0.1 is the constant c, can be modified
            v_tgt = v_teacher - t_reshape * dv_dt
            return v_pred, v_tgt
        elif self.distill_type == 'cm':
            t_tgt = self.get_target_t(t)
            t_prev = t - self.interval
            # Student model
            v_pred = self.training_models['denoiser'](x_t, t * 1000, cond_train, **kwargs)
            pred_tgt = self.euler_step(x_t, v_pred, t-t_tgt)
            # Teacher model
            with torch.no_grad():
                x_prev = self.euler_step(x_t, v_teacher, t - t_prev)
                v_prev = self.training_models['denoiser'](x_prev, t_prev * 1000, cond_train, **kwargs)
                teacher_tgt = self.euler_step(x_prev, v_prev, t_prev-t_tgt)
            return pred_tgt, teacher_tgt
        else:
            raise ValueError(f"Unknown distill_type: {self.distill_type}")
        
    def training_losses(
        self,
        x_0: torch.Tensor,
        cond=None,
        **kwargs
    ) -> Tuple[Dict, Dict]:
        """
        Compute training losses for a single timestep.

        Args:
            x_0: The [N x C x ...] tensor of noiseless inputs.
            cond: The [N x ...] tensor of additional conditions.
            kwargs: Additional arguments to pass to the backbone.

        Returns:
            a dict with the key "loss" containing a tensor of shape [N].
            may also contain other keys for different terms.
        """
        self.training_models['denoiser'].train()
        noise = torch.randn_like(x_0)
        t = self.sample_t(x_0.shape[0]).to(x_0.device).float()
        cond_train = self.get_cond(cond, **kwargs)

        terms = edict()
        loss = 0.
        # Velocity Matching Loss
        if self.lambda_vm > 0:
            v_pred, v_tgt = self.get_v_pred_tgt(x_0, noise, t, cond, self.cfg_teacher_vm, **kwargs)
            loss_vm = self.lambda_vm * torch.square(v_pred - v_tgt)
            loss = loss + loss_vm
            terms["loss_vm"] = loss_vm.mean()
        # Velocity Distillation Loss
        if self.lambda_vd > 0:
            t1 = torch.ones(x_0.shape[0]).to(x_0.device)
            noise_fake = torch.randn_like(x_0)
            v_fake = self.training_models['denoiser'](noise_fake, t1 * 1000, cond_train, **kwargs)
            x_fake = noise_fake - v_fake
            with torch.no_grad():
                v_pred, v_tgt = self.get_v_pred_tgt(x_fake, noise, t, cond, self.cfg_teacher_vd, **kwargs)
            loss_vd = self.lambda_vd * torch.square(v_pred - v_tgt + v_fake - v_fake.detach())
            loss = loss + loss_vd
            terms["loss_vd"] = loss_vd.mean()
        terms["loss"] = loss.mean()

        # log loss with time bins
        mse_per_instance = np.array([
            loss[i].mean().item()
            for i in range(x_0.shape[0])
        ])
        time_bin = np.digitize(t.cpu().numpy(), np.linspace(0, 1, 11)) - 1
        for i in range(10):
            if (time_bin == i).sum() != 0:
                terms[f"bin_{i}"] = {"loss": mse_per_instance[time_bin == i].mean()}

        return terms, {}
    
    @torch.no_grad()
    def run_snapshot(
        self,
        num_samples: int,
        batch_size: int,
        verbose: bool = False,
        seed: int = 3407, # fix img at each eval
        sample_steps: list[int] = [25, 4, 3, 2, 1]
    ) -> Dict:
        g = torch.Generator()
        g.manual_seed(seed + self.rank)
        dataloader = DataLoader(
            copy.deepcopy(self.dataset),
            batch_size=batch_size,
            shuffle=True,
            num_workers=0,
            collate_fn=self.dataset.collate_fn if hasattr(self.dataset, 'collate_fn') else None,
            generator=g,
        )

        # inference
        sampler = self.get_sampler()
        sample_gt = []
        samples = {}
        for sample_step in sample_steps:
            samples[f'N{sample_step}'] = []
        cond_vis = []
        for i in range(0, num_samples, batch_size):
            batch = min(batch_size, num_samples - i)
            with set_local_seed(seed + i*self.world_size + self.rank):
                data = next(iter(dataloader))
            data = {k: v[:batch].cuda() if isinstance(v, torch.Tensor) else v[:batch] for k, v in data.items()}
            noise = torch.randn_like(data['x_0'])
            sample_gt.append(data['x_0'])
            cond_vis.append(self.vis_cond(**data))
            del data['x_0']
            args = self.get_inference_cond(**data)
            for sample_step in sample_steps:
                res = sampler.sample(
                    self.models['denoiser'],
                    noise=noise,
                    **args,
                    steps=sample_step, cfg_strength=3.0, verbose=verbose,
                    rescale_t=self.rescale_t,
                )
                samples[f'N{sample_step}'].append(res.samples)

        sample_gt = torch.cat(sample_gt, dim=0)
        for k in samples.keys():
            samples[k] = torch.cat(samples[k], dim=0)
        sample_dict = {
            'sample_gt': {'value': sample_gt, 'type': 'sample'},
        }
        for k in samples.keys():
            sample_dict[f'sample_{k}'] = {'value': samples[k], 'type': 'sample'}
        sample_dict.update(dict_reduce(cond_vis, None, {
            'value': lambda x: torch.cat(x, dim=0),
            'type': lambda x: x[0],
        }))
        
        return sample_dict
    

class FlowMDTCFGTrainer(ClassifierFreeGuidanceMixin, FlowMDTTrainer):
    """
    Trainer for diffusion model with MDT-dist objective and classifier-free guidance.
    
    Args:
        models (dict[str, nn.Module]): Models to train.
        dataset (torch.utils.data.Dataset): Dataset.
        output_dir (str): Output directory.
        load_dir (str): Load directory.
        step (int): Step to load.
        batch_size (int): Batch size.
        batch_size_per_gpu (int): Batch size per GPU. If specified, batch_size will be ignored.
        batch_split (int): Split batch with gradient accumulation.
        max_steps (int): Max steps.
        optimizer (dict): Optimizer config.
        lr_scheduler (dict): Learning rate scheduler config.
        elastic (dict): Elastic memory management config.
        grad_clip (float or dict): Gradient clip config.
        ema_rate (float or list): Exponential moving average rates.
        fp16_mode (str): FP16 mode.
            - None: No FP16.
            - 'inflat_all': Hold a inflated fp32 master param for all params.
            - 'amp': Automatic mixed precision.
        fp16_scale_growth (float): Scale growth for FP16 gradient backpropagation.
        finetune_ckpt (dict): Finetune checkpoint.
        log_param_stats (bool): Log parameter stats.
        i_print (int): Print interval.
        i_log (int): Log interval.
        i_sample (int): Sample interval.
        i_save (int): Save interval.
        i_ddpcheck (int): DDP check interval.

        t_schedule (dict): Time schedule for flow matching.
        sigma_min (float): Minimum noise level.
        p_uncond (float): Probability of dropping conditions.
    """
    pass


class TextConditionedFlowMDTCFGTrainer(TextConditionedMixin, FlowMDTCFGTrainer):
    """
    Trainer for text-conditioned diffusion model with MDT-dist objective and classifier-free guidance.
    
    Args:
        models (dict[str, nn.Module]): Models to train.
        dataset (torch.utils.data.Dataset): Dataset.
        output_dir (str): Output directory.
        load_dir (str): Load directory.
        step (int): Step to load.
        batch_size (int): Batch size.
        batch_size_per_gpu (int): Batch size per GPU. If specified, batch_size will be ignored.
        batch_split (int): Split batch with gradient accumulation.
        max_steps (int): Max steps.
        optimizer (dict): Optimizer config.
        lr_scheduler (dict): Learning rate scheduler config.
        elastic (dict): Elastic memory management config.
        grad_clip (float or dict): Gradient clip config.
        ema_rate (float or list): Exponential moving average rates.
        fp16_mode (str): FP16 mode.
            - None: No FP16.
            - 'inflat_all': Hold a inflated fp32 master param for all params.
            - 'amp': Automatic mixed precision.
        fp16_scale_growth (float): Scale growth for FP16 gradient backpropagation.
        finetune_ckpt (dict): Finetune checkpoint.
        log_param_stats (bool): Log parameter stats.
        i_print (int): Print interval.
        i_log (int): Log interval.
        i_sample (int): Sample interval.
        i_save (int): Save interval.
        i_ddpcheck (int): DDP check interval.

        t_schedule (dict): Time schedule for flow matching.
        sigma_min (float): Minimum noise level.
        p_uncond (float): Probability of dropping conditions.
        text_cond_model(str): Text conditioning model.
    """
    pass


class ImageConditionedFlowMDTCFGTrainer(ImageConditionedMixin, FlowMDTCFGTrainer):
    """
    Trainer for image-conditioned diffusion model with MDT-dist objective and classifier-free guidance.
    
    Args:
        models (dict[str, nn.Module]): Models to train.
        dataset (torch.utils.data.Dataset): Dataset.
        output_dir (str): Output directory.
        load_dir (str): Load directory.
        step (int): Step to load.
        batch_size (int): Batch size.
        batch_size_per_gpu (int): Batch size per GPU. If specified, batch_size will be ignored.
        batch_split (int): Split batch with gradient accumulation.
        max_steps (int): Max steps.
        optimizer (dict): Optimizer config.
        lr_scheduler (dict): Learning rate scheduler config.
        elastic (dict): Elastic memory management config.
        grad_clip (float or dict): Gradient clip config.
        ema_rate (float or list): Exponential moving average rates.
        fp16_mode (str): FP16 mode.
            - None: No FP16.
            - 'inflat_all': Hold a inflated fp32 master param for all params.
            - 'amp': Automatic mixed precision.
        fp16_scale_growth (float): Scale growth for FP16 gradient backpropagation.
        finetune_ckpt (dict): Finetune checkpoint.
        log_param_stats (bool): Log parameter stats.
        i_print (int): Print interval.
        i_log (int): Log interval.
        i_sample (int): Sample interval.
        i_save (int): Save interval.
        i_ddpcheck (int): DDP check interval.

        t_schedule (dict): Time schedule for flow matching.
        sigma_min (float): Minimum noise level.
        p_uncond (float): Probability of dropping conditions.
        image_cond_model (str): Image conditioning model.
    """
    pass
