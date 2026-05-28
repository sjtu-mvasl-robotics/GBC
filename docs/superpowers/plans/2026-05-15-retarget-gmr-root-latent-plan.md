# Retarget Latent Core Design

## 目标

把当前 `angle-axis -> robot action` 的单机器人回归器，重构成一个可复用的 motion latent：

```text
human angle-axis + root pose + optional beta
    -> encoder
    -> shared latent z
    -> human decoder: reconstruct human pose/root
    -> robot decoder: reconstruct robot root/action
```

GMR 只负责离线产生高质量 robot root/action target；训练阶段直接 load 处理好的 paired dataset。

第一阶段核心是 **SFT**：用 GMR 离线 paired 数据做 supervised learning，先把 human latent、human reconstruction decoder、robot root/action decoder 跑通。不做 RLFT，不做 reward optimization，不把 flow/diffusion decoder 放进首个核心 baseline。

## 推荐默认方案

默认先做 **GMR-SFT + continuous VAE latent baseline**，再加 discrete token ablation。

理由：

- 这个阶段的核心问题是先验证 shared latent 是否能同时保留 human pose 和 robot motion，而不是一开始解决 controller/VLA tokenization。
- 连续 VAE 更容易训练、debug 和看 latent smoothness；对 root/action 重建误差更直接。
- VQ-VAE/FSQ 很适合 controller-facing token，但会引入量化瓶颈、code usage、collapse/underuse 等额外问题。
- SONIC 的 token 设计很有价值，但更适合作为第二阶段：把验证过的 continuous latent 压成 compact universal token。

结论：

```python
training_stage = "sft"                # default first stage
latent_type = "continuous_vae"        # default
latent_dim = 256                      # per-frame or per-token latent
shape_beta_dim = 16                   # optional static conditioning
shape_beta_embed_dim = 32
sequence_window = 40                  # aligns with SONIC-style token window, roughly 0.8s at 50Hz
```

保留两个 ablation：

```python
latent_type = "vq_vae"                # codebook baseline
codebook_size = 512 or 1024
code_dim = 128

latent_type = "fsq"                   # SONIC-style controller token
token_dim = 64
token_window = 40
```

## 输入表示

默认输入不要把所有东西揉成一个平铺向量。分三类信号：

```python
pose_aa:        (B, T, J, 3)          # human body angle-axis, no root
root_pose:      (B, T, 9)             # root_pos(3) + root_rot_6d(6)
beta:           (B, 16) or None       # static SMPLH shape / robot-fit beta
```

说明：

- 如果说 “root 6D pose” 指 SE(3) 六自由度，训练里仍建议用 `root_pos(3) + root_rot_6d(6)`，避免旋转表示不连续。
- `beta` 不应并入 angle-axis。angle-axis 表示动态姿态；beta 表示静态身体/拟合形状。
- beta 建议默认作为 conditioning 打开，因为 robot 拟合位置本质上来自一个特殊 SMPLH beta；没有 beta 时用 zero + mask，不能让模型误以为 zero beta 一定是真实形状。

## Encoder 结构

推荐结构：**body-part token + temporal transformer VAE encoder**。

输入先按身体部位拆分：

```text
left_arm:  shoulder, elbow, wrist
right_arm: shoulder, elbow, wrist
trunk:     pelvis, spine, chest, head
left_leg:  hip, knee, ankle, toe
right_leg: hip, knee, ankle, toe
root:      root_pos + root_rot_6d
shape:     beta embedding
```

每一帧形成若干 tokens：

```python
joint_tokens = MLP(angle_axis_joints)
root_token = MLP(root_pose)
shape_token = MLP(beta)               # repeated across time or used as prefix token
```

然后：

```python
tokens -> body_part_pooling -> temporal_transformer -> mu, logvar
z = mu + eps * exp(0.5 * logvar)
```

推荐维度：

```python
token_dim = 256
num_layers = 4 to 6
num_heads = 4 or 8
z_dim = 256
```

为什么不是简单 MLP：

- root/action 的连续性要求来自时间窗口，不是单帧关系。
- body-part token 能借鉴 UniT 的 decoupled latent 思路，避免腿部和手臂在 fused latent 里互相污染。
- shape beta 作为 prefix/static token 比拼到每帧向量里更清楚。

## Decoder 结构

### Decoder 1: Human Reconstruction Decoder

目的：让 latent 保持 human motion manifold，而不是只变成 robot action embedding。

```python
HumanDecoder(z, beta_cond) -> {
    pose_aa_recon:   (B, T, J, 3),
    root_pos_recon:  (B, T, 3),
    root_rot6d_recon:(B, T, 6),
}
```

结构：

```text
z + temporal position embedding + beta token
    -> transformer decoder / temporal transformer
    -> pose head + root head
```

Loss：

```python
L_human = angle_axis_geodesic_or_l2 + root_pos_l1 + root_rot6d_geodesic
```

### Decoder 2: Robot-Specific Decoder

第一阶段用 deterministic decoder，作为 VAE baseline：

```python
RobotDecoder(z, robot_embedding, beta_cond) -> {
    robot_root_pos:  (B, T, 3),
    robot_root_rot6d:(B, T, 6),
    actions:         (B, T, D_robot),
}
```

结构：

```text
z + robot embedding + beta token
    -> shared temporal decoder
    -> robot-specific output head
```

这里建议 **shared trunk + robot-specific heads**，不要一开始就全共享输出：

- 不同机器人 DOF 数和 joint order 不同。
- shared trunk 学通用 motion，head 处理具体构型。
- 后续可以改成 graph decoder，但 baseline 先别复杂化。

Loss：

```python
L_robot = action_l1_or_mse + root_pos_l1 + root_rot6d_geodesic + FK_link_loss + smoothness_loss
```

### 后续扩展: Robot-to-Human Encoder

后续可加：

```python
RobotEncoder(robot_root_pose, actions, robot_id) -> z
HumanDecoder(z) -> human pose/root
```

训练目标：

```python
z_h = HumanEncoder(human_pose)
z_r = RobotEncoder(robot_motion)

L_align = ||z_h - z_r||^2
L_cycle = ||HumanDecoder(z_r) - human_pose|| + ||RobotDecoder(z_h) - robot_motion||
```

这个扩展对应 SONIC/UniT 的 shared latent alignment，但不建议第一阶段就阻塞主线。

## Continuous VAE vs VQ-VAE vs FSQ

### 方案 A: Continuous VAE Baseline

推荐默认。

```python
q(z | human_pose, root, beta) = N(mu, sigma)
z_dim = 256
```

优点：

- 最容易训练。
- latent 可插值、可平滑。
- 重建 root/action 的误差最直观。
- 更适合先验证 GMR-SFT 数据是否真的能学到比当前 PoseTransformer 更好的 mapping。

缺点：

- 不是天然 controller/VLA token。
- latent 的下游使用形态不如 SONIC 那样离散稳定。

### 方案 B: VQ-VAE

作为第二个 baseline。

```python
codebook_size = 512 or 1024
code_dim = 128
tokens_per_window = T
```

优点：

- 可以产生离散 motion token。
- 方便和 planning / language model 结合。

缺点：

- codebook usage 是额外问题。
- retargeting 数据如果不够多样，容易 underuse。
- early stage 会干扰判断 encoder/decoder 结构是否正确。

### 方案 C: SONIC-style FSQ

作为 controller-facing token 方案。

```python
token_dim = 64
token_window = 40
quantizer = FSQ
```

优点：

- SONIC 已经证明这种 compact universal token 适合给 VLA/teleop/controller 使用。
- FSQ 没有传统 VQ codebook collapse 问题。
- 输出形态清楚：上游模型预测 token，下游 decoder/control policy 执行。

缺点：

- 对当前 retargeting 训练来说不是最小可行方案。
- 如果 robot decoder 还没学好，离散化会让问题更难定位。

推荐顺序：

```text
Stage 1: GMR-SFT continuous VAE latent
Stage 2: VQ-VAE / FSQ token ablation
Stage 3: conditional flow/denoising robot decoder if deterministic SFT decoder is too averaged
Stage 4: SONIC-style 64-dim token form for controller/VLA
Stage 5: RLFT / simulator-in-loop refinement
```

## SONIC 给我们的启发

SONIC 的关键不是 “用了离散 token” 这么简单，而是：

```text
human encoder
robot encoder
hybrid/sparse command encoder
    -> universal token
    -> control decoder
    -> motion command reconstruction decoder
```

对这个 retarget model 的启发：

- latent 不能只服务 robot action 重建，也要能重建 motion command，否则很容易变成某个机器人专用 action embedding。
- 需要 human token 和 robot token 对齐：`L_token = ||z_h - z_r||`。
- 需要 cycle：`human -> token -> robot motion -> token`，保证跨 embodiment 翻译后 latent 不漂。
- 最终若要接 VLA/MUJICA，建议导出 `64-dim x 40` 这类 compact token。

但第一阶段不要直接照搬 SONIC 的控制器训练。我们现在做的是 retargeting model，不是全身 tracking policy。

## UniT 给我们的启发

UniT 更适合作为 latent 结构启发：

- body-part decoupled latent：左右手、躯干、左右腿拆开。
- 不同 embodiment 通过共享 latent 对齐，而不是强行共享 action head。
- cross-reconstruction 比单纯 latent MSE 更有用。

对本设计的落地：

```python
z = {
    "left_arm":  z_la,
    "right_arm": z_ra,
    "trunk":     z_tr,
    "left_leg":  z_ll,
    "right_leg": z_rl,
    "global":    z_global,
}
```

最后可以 concat/project 成 `z_dim=256`，但 loss 应该保留 part-level 监督，尤其手脚 contact/end-effector。

## 其它模型给出的取舍

### GMR

GMR 是离线 target generator；训练阶段不调用 GMR。

它给的数据应该包含：

```python
robot_root_pos
robot_root_rot
robot_actions
valid_mask
```

这正好支持 robot decoder 的 root/action 监督。

### 当前 SMPLH Fit 是否要优化

结论：**可行，且针对 foot contact 问题有价值；但要做成 fit/data-prep 的局部增强，不要把它变成第一阶段 SFT 模型本身的复杂依赖。**

当前 `SMPLHFitter` 做的是 joint-only fitting：

```text
optimize: beta, dmpls, global scale
loss: selected SMPLH joints <-> mapped URDF links
optional: T-pose / additional pose joint losses
```

它没有直接优化 mesh surface、body-part volume、foot/contact anchor、segment thickness，也没有学习 per-link offset。脚尖尤其明显：SMPLH 有 mesh surface，但没有一个天然等价于 robot toe/contact patch 的 joint；URDF 里也常常是 ankle/foot link 而不是 toe joint。所以只用 joint-level fitting 时，contact anchor 一定会有系统误差。

对当前 latent 方案的影响主要是：

- `robot_fit_beta` 会作为 shape conditioning 输入 latent；如果 fit 有系统偏差，beta token 会把偏差传给 robot decoder。
- FK/link loss 会受 fit quality 影响；尤其 foot、wrist、pelvis/root 附近的误差会影响 contact 和 root pose 学习。
- 第一阶段 SFT 的主要监督来自 GMR 离线 robot root/action，因此可以先用现有 fit 作为 baseline；但如果 contact 监督要进入 loss 或 dataset metrics，建议加一个轻量 mesh-anchor fit。

第一阶段建议做三个轻量动作：

```python
save robot_fit_beta, dmpls, scale into paired dataset
log fit diagnostics: joint_error, end_effector_error, segment_length_error, foot_anchor_error
precompute optional surface anchors: foot/toe, wrist/hand, pelvis/trunk
```

可行的 mesh-anchor 方案：

```text
SMPLH side:
  use body_model output vertices/faces
  select body-part vertex groups around foot sole/toe, heel, wrist/hand, pelvis
  sample or average local surface points

Robot side:
  parse URDF visual/collision meshes per link
  sample local points from foot/ankle/hand/base links
  transform points through FK for each fitting pose

Loss:
  joint_loss
  + segment_length_loss
  + foot_surface_anchor_loss
  + wrist_surface_anchor_loss
  + local Chamfer or point-to-anchor loss
  + beta/scale/small_link_offset prior
```

这里不要做全身 symmetric Chamfer。全身 Chamfer 容易把 torso/limb volume 拟合过头，而且 robot mesh 和 human mesh 形状差异太大。更合理的是 **局部对应点回归**：

```text
human toe/sole surface patch <-> robot foot front/sole patch
human heel patch <-> robot heel/back-foot patch
human palm/wrist patch <-> robot wrist/end-effector patch
human pelvis/trunk anchors <-> robot base/torso anchors
```

对 contact 最有帮助的是 foot patch，而不是 ankle joint。contact label 和 foot slip loss 都应该优先使用这些 sampled foot surface anchors。

Robot mesh 对应点来源和梯度路径：

```text
1. Robot points come from URDF link geometry.
   Prefer collision mesh for contact anchors.
   Fall back to visual mesh if collision mesh is missing.
   Fall back to primitive box/cylinder or manually configured local anchors if no mesh exists.

2. Correspondence is semantic and fixed, not dynamic nearest-neighbor.
   human foot sole/toe patch point at normalized (u, v)
       <-> robot foot sole/front patch point at same normalized (u, v)
   human heel patch
       <-> robot heel/back-foot patch
   human wrist/palm patch
       <-> robot wrist/end-effector patch

3. Store correspondences as local differentiable samples.
   human: SMPLH face indices + barycentric weights
   robot: link name + local mesh point coordinates or mesh face indices + barycentric weights

4. Training-time forward is fully torch.
   human_point = barycentric_sample(body_model(...).v, face_ids, bary_weights)
   robot_point = FK(action)[link_name] @ robot_local_point
   loss = robust_l1(human_point - robot_point)
```

不要在 forward 里用 `trimesh`、`numpy` 或动态 `argmin nearest neighbor` 重新匹配点。`trimesh` 只用于离线读取 mesh、采样局部 patch、保存固定 local anchors。训练时的梯度来自：

```text
human_point -> body_model vertices -> beta / pose / scale
robot_point -> torch FK transform -> action / root / optional link offset
loss -> both point sets
```

如果局部 patch 的点序不可靠，可以用 soft correspondence，但也要限制在同一个 semantic patch 内：

```python
weights = softmax(-cdist(human_patch_points, robot_patch_points) / tau)
loss = sum_ij weights_ij * ||human_i - robot_j||^2
```

这比 hard nearest-neighbor 更平滑，但第一版更推荐固定 `(u, v)` patch correspondences，因为它稳定、可解释，也更容易 debug contact anchor。

触发优化的条件：

```text
mean end-effector fit error > 2-3 cm
foot/toe anchor error causes contact labels or FK foot positions to drift
G1/H1-2 shared latent alignment improves poorly unless beta conditioning is removed
robot decoder root/action loss has consistent body-part bias
```

如果触发，再做更完整的 `mesh-aware SMPLH fit`：

```text
joint loss
+ segment length loss
+ foot/wrist anchor loss
+ sampled robot mesh/body-part Chamfer loss
+ beta/scale/offset priors
```

默认仍不要优化自由 per-vertex mesh residual；先优化 `beta + scale + small link offsets`。per-vertex residual 只适合作为最后诊断项，否则很容易把 SMPLH 变成不可解释的 robot-specific mesh。

### MUJICA / MDM / Flow Matching

它们更适合启发 robot decoder 的生成形式，而不是第一版 latent。

建议：

- VAE baseline 先用 deterministic robot decoder。
- 如果 deterministic decoder 出现 average motion、foot contact 模糊、root transition 不稳，再引入 conditional flow matching decoder。
- Flow decoder 输入条件就是 `z + robot_embedding + beta`，目标是 denoise `robot_root + action`。

### Diffusion/Flow Decoder 的定位

不是第一阶段必须项，但很可能是第二阶段提升质量的关键：

```text
continuous latent z
    -> conditional flow/denoising robot decoder
    -> smoother and multimodal robot root/action
```

这和 `../GBC_MUJICA` 有良性联动，因为 MUJICA 已有 flow matching/BPE 训练路径。

## 最小可行训练目标

第一版是 SFT，只需要这些 supervised losses：

```python
L_sft = (
    L_human_recon
    + L_robot_action
    + L_robot_root
    + L_fk_link
    + L_smooth
    + lambda_kl * KL(q(z|x) || N(0, I))
)
```

其中 `L_robot_action` 和 `L_robot_root` 都直接监督到 GMR 离线数据；`KL` 只是 continuous VAE 的 latent regularizer，不是 RL 项。

如果有多机器人 paired 数据，再加：

```python
L_align = ||z_human - z_robot||^2
L_cycle = reconstruction after human <-> robot latent translation
```

第一版不要加太多辅助项，也不要加 RLFT。先回答三个问题：

1. `z_dim=256` 的连续 VAE latent 能否同时重建 human pose/root 和 GMR robot root/action？
2. 加 beta conditioning 是否显著降低 FK/link error？
3. shared trunk + robot-specific head 是否能支持至少 G1/H1-2 两种构型？

如果这三个成立，再进入 discrete token、flow decoder 和后续 RLFT。

## 当前推荐配置

```python
model = RetargetVAE(
    training_stage="sft",
    pose_dim=21 * 3,
    root_dim=9,
    beta_dim=16,
    beta_embed_dim=32,
    z_dim=256,
    window_size=40,
    encoder="body_part_temporal_transformer",
    human_decoder="temporal_transformer",
    robot_decoder="shared_temporal_transformer_with_robot_heads",
    latent_type="continuous_vae",
)
```

第二阶段配置：

```python
latent_type="fsq"
token_dim=64
token_window=40
robot_decoder="conditional_flow_transformer"
```

一句话总结：

> 第一阶段先做 GMR-SFT，不做 RLFT：用连续 VAE 做 retarget latent baseline，`z_dim=256`，输入是 `angle-axis + root_pos/root_rot6d + beta token`，双 decoder supervised 重建 human 和 GMR robot root/action；SONIC 启发我们后续把这个 latent 压成 64-dim universal token，UniT 启发我们按 body part 对齐 latent，MUJICA/flow matching 启发我们把第二阶段高质量 robot decoder 做成 denoising/flow model，RLFT 再放到最后。
