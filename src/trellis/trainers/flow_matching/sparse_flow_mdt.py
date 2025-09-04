from typing import *
import os
import copy
import functools
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
from easydict import EasyDict as edict

from ...modules import sparse as sp
from ...utils.general_utils import dict_reduce
from ...utils.data_utils import cycle, BalancedResumableSampler, set_local_seed
from .flow_matching_mdt import FlowMDTTrainer
from .mixins.classifier_free_guidance import ClassifierFreeGuidanceMixin
from .mixins.text_conditioned import TextConditionedMixin
from .mixins.image_conditioned import ImageConditionedMixin


class SparseFlowMDTTrainer(FlowMDTTrainer):
    """
    Trainer for sparse diffusion model with MDT-dist objective.
    
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
    
    def prepare_dataloader(self, **kwargs):
        """
        Prepare dataloader.
        """
        self.data_sampler = BalancedResumableSampler(
            self.dataset,
            shuffle=True,
            batch_size=self.batch_size_per_gpu,
        )
        self.dataloader = DataLoader(
            self.dataset,
            batch_size=self.batch_size_per_gpu,
            num_workers=int(np.ceil(os.cpu_count() / torch.cuda.device_count())),
            pin_memory=True,
            drop_last=True,
            persistent_workers=True,
            collate_fn=functools.partial(self.dataset.collate_fn, split_size=self.batch_split),
            sampler=self.data_sampler,
        )
        self.data_iterator = cycle(self.dataloader)

    def get_v_pred_tgt(
        self,
        x_0: sp.SparseTensor,
        noise: sp.SparseTensor,
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
                def model_wrapper(x_t_feats, t):
                    x_t_clone = x_t.replace(x_t_feats)
                    return self.training_models['denoiser'](x_t_clone, t * 1000, cond=cond_train, **kwargs)
                # torch.autograd.functional supports reverse mode, while torch.func only supports forward mode
                _, dv_dt_feats = torch.autograd.functional.jvp(
                    model_wrapper, (x_t.feats, t), (v_teacher, torch.ones_like(t))
                )
                dv_dt = x_t.replace(dv_dt_feats)
        elif self.distill_type == 'discrete':
            with torch.no_grad():
                t_prev = t - self.interval
                x_prev = self.euler_step(x_t, v_teacher, t - t_prev)
                v_prev = self.training_models['denoiser'](x_prev, t_prev * 1000, cond_train, **kwargs)
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
        # normalization
        if self.d_norm:
            dv_dt_norm = torch.cat([torch.linalg.vector_norm(dv_dt.feats[dv_dt.layout[i]], dim=(0, 1), keepdim=True).repeat(dv_dt.layout[i].stop-dv_dt.layout[i].start, dv_dt.feats.shape[1]) for i in range(dv_dt.shape[0])], dim=0)
            dv_dt_norm = dv_dt.replace(dv_dt_norm)
            dv_dt = dv_dt / (dv_dt_norm + 0.1)  # 0.1 is the constant c, can be modified
        v_tgt = v_teacher - t_reshape * dv_dt

        return v_pred, v_tgt

    def training_losses(
        self,
        x_0: sp.SparseTensor,
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
        noise = x_0.replace(torch.randn_like(x_0.feats))
        t = self.sample_t(x_0.shape[0]).to(x_0.device).float()
        cond_train = self.get_cond(cond, **kwargs)

        terms = edict()
        loss = 0.
        # Velocity Matching Loss
        if self.lambda_vm > 0:
            v_pred, v_tgt = self.get_v_pred_tgt(x_0, noise, t, cond, self.cfg_teacher_vm, **kwargs)
            mse_vm = v_pred - v_tgt
            loss_vm = self.lambda_vm * mse_vm * mse_vm
            loss = loss + loss_vm
            terms["loss_vm"] = loss_vm.feats.mean()
        # Velocity Distillation Loss
        if self.lambda_vd > 0:
            t1 = torch.ones(x_0.shape[0]).to(x_0.device)
            noise_fake = x_0.replace(torch.randn_like(x_0.feats))
            v_fake = self.training_models['denoiser'](noise_fake, t1 * 1000, cond_train, **kwargs)
            x_fake = noise_fake - v_fake
            with torch.no_grad():
                v_pred, v_tgt = self.get_v_pred_tgt(x_fake, noise, t, cond, self.cfg_teacher_vd, **kwargs)
            mse_vd = v_pred - v_tgt + v_fake - v_fake.detach()
            loss_vd = self.lambda_vd * mse_vd * mse_vd
            loss = loss + loss_vd
            terms["loss_vd"] = loss_vd.feats.mean()
        terms["loss"] = loss.feats.mean()

        # log loss with time bins
        mse_per_instance = np.array([
            loss.feats[x_0.layout[i]].mean().item()
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
        sample_steps: list[int] = [25, 4, 3, 2, 1],
    ) -> Dict:
        def seed_worker(worker_seed):
            worker_seed = seed + self.rank
            import random
            random.seed(worker_seed)
            np.random.seed(worker_seed)
        g = torch.Generator()
        g.manual_seed(seed + self.rank)
        dataloader = DataLoader(
            copy.deepcopy(self.dataset),
            batch_size=batch_size,
            shuffle=True,
            num_workers=0,
            collate_fn=self.dataset.collate_fn if hasattr(self.dataset, 'collate_fn') else None,
            generator=g,
            worker_init_fn=seed_worker,
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
            data = {k: v[:batch].cuda() if not isinstance(v, list) else v[:batch] for k, v in data.items()}
            noise = data['x_0'].replace(torch.randn_like(data['x_0'].feats))
            sample_gt.append(data['x_0'])
            cond_vis.append(self.vis_cond(**data))
            del data['x_0']
            args = self.get_inference_cond(**data)
            for sample_step in sample_steps:
                res = sampler.sample(
                    self.models['denoiser'],
                    noise=noise,
                    **args,
                    steps=sample_step, cfg_strength=0.1, verbose=verbose,
                    rescale_t=self.rescale_t,
                )
                samples[f'N{sample_step}'].append(res.samples)

        sample_gt = sp.sparse_cat(sample_gt)
        for k in samples.keys():
            samples[k] = sp.sparse_cat(samples[k], dim=0)
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


class SparseFlowMDTCFGTrainer(ClassifierFreeGuidanceMixin, SparseFlowMDTTrainer):
    """
    Trainer for sparse diffusion model with MDT-dist objective and classifier-free guidance.
    
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


class TextConditionedSparseFlowMDTCFGTrainer(TextConditionedMixin, SparseFlowMDTCFGTrainer):
    """
    Trainer for sparse text-conditioned diffusion model with MDT-dist objective and classifier-free guidance.
    
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


class ImageConditionedSparseFlowMDTCFGTrainer(ImageConditionedMixin, SparseFlowMDTCFGTrainer):
    """
    Trainer for sparse image-conditioned diffusion model with MDT-dist objective and classifier-free guidance.
    
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
