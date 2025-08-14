from dataclasses import MISSING
from typing import Literal

from isaaclab.utils import configclass


@configclass
class RslRlRefPpoActorCriticCfg:
    class_name: str = MISSING
    max_len: int = MISSING
    dim_model: int = MISSING
    num_layers: int = MISSING
    num_heads: int = MISSING
    init_noise_std: float = MISSING
    load_dagger: bool = False
    load_dagger_path: str | None = None
    load_actor_path: str | None = None
    apply_mlp_residual: bool = False
    history_length: int = 1
    concatenate_term_names: dict[str, list[list[str]]] | None = None
    concatenate_ref_term_names: dict[str, list[list[str]]] | None = None

@configclass
class RslRlRefPpoAmpNetCfg:
    backbone_input_dim: int = MISSING
    backbone_output_dim: int = MISSING
    backbone: str = "mlp"
    activation: str = "elu"
    out_activation: str = "tanh"
    net_kwargs: dict | None = None

@configclass
class RslRlPpoAmpCfg:
    net_cfg: RslRlRefPpoAmpNetCfg = RslRlRefPpoAmpNetCfg()
    learning_rate: float = 1e-3
    amp_obs_extractor: callable = MISSING
    amp_ref_obs_extractor: callable = MISSING
    epsilon: float = 1e-4
    amp_reward_scale: float = 1.0
    gradient_penalty_coeff: float = 10.0
    amp_update_interval: int = 10
    amp_pretrain_steps: int = 50


@configclass
class RslRlRefPpoAlgorithmCfg:
    class_name: str = MISSING
    value_loss_coef: float = MISSING
    clip_param: float = MISSING
    use_clipped_value_loss: bool = MISSING
    desired_kl: float = MISSING
    entropy_coef: float = MISSING
    gamma: float = MISSING
    lam: float = MISSING
    max_grad_norm: float = MISSING
    learning_rate: float = MISSING
    normalize_advantage_per_mini_batch: bool = False
    max_lr: float = 1e-2
    min_lr: float = 1e-4
    max_lr_after_certain_epoch: float = 1e-3
    max_lr_restriction_epoch: int = 5000
    min_lr_after_certain_epoch: float = 1e-5
    min_lr_restriction_epoch: int = 5000
    
    # beginning of MMPPO config
    num_learning_epochs: int = MISSING
    num_mini_batches: int = MISSING
    schedule: str = MISSING
    teacher_coef: float | None = None
    teacher_coef_range: tuple[float, float] | None = None
    teacher_coef_decay: float | None = None
    teacher_coef_decay_interval: int = 100
    teacher_coef_mode: str = "kl" # "kl" or "norm"
    teacher_loss_coef: float | None = None
    teacher_loss_coef_range: tuple[float, float] | None = None
    teacher_loss_coef_decay: float | None = None
    teacher_loss_coef_decay_interval: int = 100
    teacher_update_interval: int = 1
    teacher_supervising_intervals: int = 0
    teacher_updating_intervals: int = 0
    teacher_lr: float = 5e-4
    teacher_only_interval: int = 0
    rnd_cfg: dict | None = None
    symmetry_cfg: dict | None = None
    amp_cfg: RslRlPpoAmpCfg | None = None
@configclass
class RslRlRefOnPolicyRunnerCfg:
    seed: int = 42
    device: str = "cuda:0"
    num_steps_per_env: int = MISSING
    max_iterations: int = MISSING
    empirical_normalization: bool = MISSING
    policy: RslRlRefPpoActorCriticCfg = RslRlRefPpoActorCriticCfg()
    algorithm: RslRlRefPpoAlgorithmCfg = RslRlRefPpoAlgorithmCfg()
    save_interval: int = MISSING
    experiment_name: str = MISSING
    run_name: str = ""
    logger: Literal["tensorboard", "neptune", "wandb"] = "tensorboard"
    resume: bool = False
    load_run: str = ".*"
    load_checkpoint: str = "model_.*.pt"
    abs_checkpoint_path: str = ""