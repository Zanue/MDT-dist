# Few-step Flow for 3D Generation via Marginal-Data Transport Distillation
### [Project Page](https://zanue.github.io/mdt-dist) | [Arxiv Paper](https://arxiv.org/abs/2509.04406)

[Zanwei Zhou](https://github.com/Zanue)<sup>1,* </sup>, [Taoran Yi](https://taoranyi.com/)<sup>2,*</sup>, [Jiemin Fang](https://jaminfong.cn/)<sup>3,&dagger;</sup>, [Chen Yang](https://chensjtu.github.io/)<sup>3</sup>, [Lingxi Xie](http://lingxixie.com/Home.html)<sup>3</sup>, [Xinggang Wang](https://xwcv.github.io/)<sup>2</sup>, [Wei Shen](https://shenwei1231.github.io/)<sup>1,&dagger;</sup>, [Qi Tian](https://www.qitian1987.com/)<sup>3</sup>

<sup>1</sup>Shanghai Jiao Tong University &emsp;<sup>2</sup>Huazhong University of Science and Technology &emsp; <sup>3</sup>Huawei Inc. &emsp; 

<sup>*</sup>Equal contribution &nbsp; <sup>&dagger;</sup>Corresponding author</p>


Flow-based 3D generation models typically require dozens of sampling steps during inference. 
Though few-step distillation methods, particularly Consistency Models (CMs), have achieved substantial advancements in accelerating 2D diffusion models, they remain under-explored for more complex 3D generation tasks. 
In this study, we propose a novel framework, MDT-dist, for few-step 3D flow distillation. 
Our approach is built upon a primary objective: distilling the pretrained model to learn the Marginal-Data Transport. 
Directly learning this objective needs to integrate the velocity fields, while this integral is intractable to be implemented. Therefore, we propose two optimizable objectives, Velocity Matching (VM) and Velocity Distillation (VD), to equivalently convert the optimization target from the transport level to the velocity and the distribution level respectively. 
Velocity Matching (VM) learns to stably match the velocity fields between the student and the teacher, but inevitably provides biased gradient estimates. 
Velocity Distillation (VD) further enhances the optimization process by leveraging the learned velocity fields to perform probability density distillation.
When evaluated on the pioneer 3D generation framework TRELLIS, our method reduces sampling steps of each flow transformer from 25 to 1–2, achieving 0.68s (1 step x 2) and 0.94s (2 steps x 2) latency with 9.0x and 6.5x speedup on A800, while preserving high visual and geometric fidelity. 
Extensive experiments demonstrate that our method significantly outperforms existing CM distillation methods, and enables TRELLIS to achieve superior performance in few-step 3D generation. 

## Updates
- 9/5/2025: Release training code.

## TODO List
- Project page is under construction
- Checkpoints

## Citation
If you find this repository/work helpful in your research, welcome to cite the paper and give a ⭐.
Some source code of ours is borrowed from [TRELLIS](https://github.com/Microsoft/TRELLIS). We sincerely appreciate the excellent works of these authors.
```
@article{mdt-dist,
    title={Few-step Flow for 3D Generation via Marginal-Data Transport Distillation},
    author={Zhou, Zanwei and Yi, Taoran and Fang, Jiemin and Yang, Chen and Xie, Lingxi and Wang, Xinggang and Shen, Wei and Tian, Qi}
    journal={arXiv:2509.04406},
    year={2025}
}
```
