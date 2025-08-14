import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Dict, List, Optional
class PoseTransformer(nn.Module):
    '''
    Pose Transformer

    Transformer for mapping SMPL pose (angle-axis) to humanoid pose (actions for given DOFs)

    Args:
        num_joints: Number of joints in the SMPL model
        input_dim: Dimension of the input pose vector
        num_actions: Number of actions for the humanoid model
        embedding_dim: Dimension of the embedding
        num_heads: Number of attention heads
        num_layers: Number of transformer layers

    Shape:
        - Input: (N, num_joints, input_dim)
        - Output: (N, num_actions)
    '''
    def __init__(self, 
                 load_hands: bool = False,
                 num_actions: int = 29,
                 embedding_dim: int = 64,
                 joint_embedding_dim: int = 256,
                 num_heads: int = 4,
                 num_layers: int = 4
                ):
        super(PoseTransformer, self).__init__()
        self.num_joints = 21 if not load_hands else 51
        self.input_dim = 3
        self.num_actions = num_actions
        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        self.num_layers = num_layers

        self.joint_embedding = nn.Linear(self.num_joints, joint_embedding_dim)
        self.embedding = nn.Linear(self.input_dim, embedding_dim)
        self.positional_encoding = nn.Parameter(torch.zeros(1, joint_embedding_dim, embedding_dim))
        encoder = nn.TransformerEncoderLayer(d_model=embedding_dim, nhead=num_heads, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder, num_layers=num_layers)
        self.decoder = nn.Linear(embedding_dim, num_actions)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        '''
        Forward pass

        Args:
            x: Input pose tensor (N, num_joints * input_dim)

        Returns:
            Output action tensor (N, num_actions)
        '''
        N = x.shape[0]
        
        x = x.view(N, self.num_joints, self.input_dim)
        x = x.permute(0, 2, 1)
        x = self.joint_embedding(x)
        x = x.permute(0, 2, 1)
        x = self.embedding(x) 
        x += self.positional_encoding
        x = self.transformer(x)
        x = x.mean(dim=1)
        x = self.decoder(x)
        return x
    
    def load_state_dict(self, state_dict, strict = True, assign = False):
        if 'model' in state_dict.keys():
            return super().load_state_dict(state_dict["model"], strict, assign)
        return super().load_state_dict(state_dict, strict, assign)
    
    
def drop_path(x, drop_prob: float = 0., training: bool = False):
    """
    Drop connections (Stochastic Depth) per sample.
    """
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()
    output = x.div(keep_prob) * random_tensor
    return output

class DropPath(nn.Module):
    """Drop connections (Stochastic Depth)."""
    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)

class LayerScale(nn.Module):
    """
    LayerScale from CaiT, used in modern ViTs to stabilize deep networks.
    """
    def __init__(self, dim, init_values=1e-5, inplace=False):
        super().__init__()
        self.inplace = inplace
        self.gamma = nn.Parameter(init_values * torch.ones(dim))

    def forward(self, x):
        return x.mul_(self.gamma) if self.inplace else x * self.gamma


# --- ViT-Enhanced Transformer Encoder Layer ---

class ViTEncoderLayer(nn.Module):
    """
    A standard Transformer Encoder Layer enhanced with LayerScale and DropPath.
    """
    def __init__(self, d_model: int, nhead: int, dim_feedforward: int = 2048, 
                 dropout: float = 0.1, drop_path: float = 0., 
                 layer_scale_init_value: float = 1e-5):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
        # --- ViT Enhancements ---
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.ls1 = LayerScale(d_model, layer_scale_init_value)
        self.ls2 = LayerScale(d_model, layer_scale_init_value)
        self.activation = F.gelu

    def forward(self, src: torch.Tensor) -> torch.Tensor:
        # Self-attention part with LayerScale and DropPath
        attn_output, _ = self.self_attn(src, src, src)
        src = src + self.drop_path(self.ls1(attn_output))
        src = self.norm1(src)
        
        # Feedforward part with LayerScale and DropPath
        ffn_output = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.drop_path(self.ls2(ffn_output))
        src = self.norm2(src)
        return src

# --- The Final, ViT-Enhanced Original Model ---

class PoseViT(nn.Module):
    """
    This is your original, successful architecture, now enhanced with modern
    ViT stabilization techniques (DropPath and LayerScale) for improved
    trainability and performance.

    Args:
        drop_path_rate (float): Stochastic depth rate. If > 0, applies DropPath.
        layer_scale_init_value (float): Initial value for LayerScale.
        ... (other parameters from your original model)
    """
    def __init__(self,
                 load_hands: bool = False,
                 robot_actions: int = 29,
                 spatial_feature_dim: int = 256,
                 d_model: int = 128,
                 nhead: int = 8,
                 num_layers: int = 2,
                 dropout: float = 0.1,
                 # --- New ViT Parameters ---
                 drop_path_rate: float = 0.1,
                 layer_scale_init_value: float = 1e-5):
        super().__init__()
        
        self.num_body_joints = 21
        self.num_hand_joints = 30
        self.num_total_joints = self.num_body_joints + self.num_hand_joints if load_hands else self.num_body_joints
        self.input_dim = 3

        # --- Stage 1: Your Proven Spatial Feature Extractor ---
        self.spatial_feature_extractor = nn.Linear(self.num_total_joints, spatial_feature_dim)
        
        # --- Stage 2: ViT-Enhanced Transformer Backend ---
        self.feature_embedder = nn.Linear(self.input_dim, d_model)
        self.pos_embed = nn.Parameter(torch.randn(1, spatial_feature_dim, d_model))
        
        # Create a linear ramp for the drop path rate for each layer
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, num_layers)]
        
        # Use the new ViT-enhanced encoder layer
        encoder_layers = [
            ViTEncoderLayer(
                d_model=d_model, nhead=nhead, dim_feedforward=d_model*4,
                dropout=dropout, drop_path=dpr[i], 
                layer_scale_init_value=layer_scale_init_value
            ) for i in range(num_layers)
        ]
        self.transformer_encoder = nn.Sequential(*encoder_layers)
        
        # --- Stage 3: Your Proven Decoder Head ---
        self.decoder_head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, robot_actions)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        N = x.shape[0]
        x = x.view(N, self.num_total_joints, self.input_dim)
        
        x_permuted = x.permute(0, 2, 1)
        spatial_features = self.spatial_feature_extractor(x_permuted)
        
        x_new_sequence = spatial_features.permute(0, 2, 1)
        x_embedded = self.feature_embedder(x_new_sequence)
        
        x_with_pos = x_embedded + self.pos_embed
        transformer_output = self.transformer_encoder(x_with_pos)
        
        global_feature = transformer_output.mean(dim=1)
        actions = self.decoder_head(global_feature)
        
        return actions


class GatedMLP(nn.Module):
    def __init__(self, in_features: int, hidden_features: int, out_features: int, dropout: float = 0.1):
        super().__init__()
        self.fc1 = nn.Linear(in_features, hidden_features * 2)
        self.activation = nn.SiLU()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.fc1(x)
        gate, activated_val = y.chunk(2, dim=-1)
        gated_val = self.activation(gate) * activated_val
        return self.dropout(self.fc2(gated_val))

class MoELayer(nn.Module):
    def __init__(self, in_features: int, num_experts: int, top_k: int, expert_hidden_scale: int = 4, dropout: float = 0.1):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        self.gate = nn.Linear(in_features, num_experts, bias=False)
        self.experts = nn.ModuleList([
            GatedMLP(in_features, in_features * expert_hidden_scale, in_features, dropout) 
            for _ in range(num_experts)
        ])

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size, seq_len, in_features = x.shape
        x_flat = x.view(-1, in_features)
        gate_logits = self.gate(x_flat)
        top_k_weights, top_k_indices = torch.topk(gate_logits, self.top_k, dim=-1)
        top_k_weights = F.softmax(top_k_weights, dim=-1, dtype=torch.float).to(x.dtype)
        
        y_flat = torch.zeros_like(x_flat)
        dispatch_mask = F.one_hot(top_k_indices, num_classes=self.num_experts).permute(2, 0, 1)
        expert_inputs = torch.einsum('b d, e b k -> e k b d', x_flat, dispatch_mask)
        expert_outputs = [self.experts[i](expert_inputs[i].squeeze(0)) for i in range(self.num_experts)]
        y_flat = torch.einsum('e k b d, e b k -> b d', torch.stack(expert_outputs), dispatch_mask * top_k_weights.unsqueeze(0))
        y = y_flat.view(batch_size, seq_len, in_features)

        fraction_of_tokens_per_expert = dispatch_mask.sum(dim=[1,2]) / len(x_flat)
        mean_prob_per_expert = gate_logits.softmax(dim=-1).mean(dim=0)
        aux_loss = self.num_experts * torch.sum(fraction_of_tokens_per_expert * mean_prob_per_expert)
        return y, aux_loss

class MoETransformerEncoderLayer(nn.Module):
    def __init__(self, d_model: int, nhead: int, num_experts: int, top_k: int, dropout: float = 0.1):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.moe = MoELayer(d_model, num_experts, top_k, dropout=dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, src: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        attn_output, _ = self.self_attn(src, src, src)
        src = src + self.dropout(attn_output)
        src = self.norm1(src)
        moe_output, aux_loss = self.moe(src)
        src = src + self.dropout(moe_output)
        src = self.norm2(src)
        return src, aux_loss

class PoseTransformerV2(nn.Module):
    def __init__(self, load_hands=False, robot_actions=29, 
                 d_model=256, nhead=8, num_layers=4, dropout=0.1):
        super().__init__()
        self.num_joints = 21 if not load_hands else 51
        self.input_dim = 3
        
        self.joint_embedder = nn.Linear(self.input_dim, d_model)
        self.pos_embed = nn.Parameter(torch.randn(1, self.num_joints + 1, d_model))
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model))
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=d_model*4, 
            dropout=dropout, batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        self.decoder_head = nn.Sequential(
            nn.LayerNorm(d_model),
            GatedMLP(d_model, d_model * 2, robot_actions)
        )

    def forward(self, x):
        N = x.shape[0]
        x = x.view(N, self.num_joints, self.input_dim)
        
        x = self.joint_embedder(x)
        
        cls_tokens = self.cls_token.expand(N, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embed
        
        x = self.transformer_encoder(x)
        
        feature = x[:, 0]
        return self.decoder_head(feature)
