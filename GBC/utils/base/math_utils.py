from copy import deepcopy
from typing import List, Literal, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.spatial import KDTree
from scipy.signal import savgol_filter

def list_expand(input: List, times: int, length: int) -> List:
    """
    Expand a list by repeating it the specified number of times.

    Args:
        input (List): The list to expand.
    """
    result = []
    for i in range(times):
        extended = [x + i * length for x in input]
        result.extend(extended)
    return result


def hampel_filter(x: torch.Tensor, window_size: int = 3, n_sigma: float = 3.0):
    """
    Apply Hampel filter to a 2D tensor (T, D) along the time axis (T).
    The Hampel filter is a robust outlier detector and remover that uses median and MAD.

    Args:
        x: Input tensor of shape (T, D)
        window_size: Half window size for the filter
        n_sigma: Number of standard deviations to use for the threshold

    Returns:
        corrected_x: Tensor of shape (T, D) after Hampel filtering
        has_outlier: Boolean indicating whether any outlier was found
    """
    T, D = x.shape
    corrected_x = x.clone()
    has_outlier = False

    # For each time step
    for t in range(T):
        # Calculate window boundaries with clamping
        t_start = max(0, t - window_size)
        t_end = min(T, t + window_size + 1)

        # Extract window
        window = x[t_start:t_end]  # shape: (window_size, D)

        # Calculate median of the window for each dimension
        median = torch.median(window, dim=0).values  # shape: (D,)

        # Calculate MAD (Median Absolute Deviation)
        mad = torch.median(torch.abs(window - median.unsqueeze(0)), dim=0).values  # shape: (D,)

        # Calculate threshold (1.4826 is used to make MAD consistent with Gaussian distribution)
        threshold = n_sigma * mad * 1.4826  # shape: (D,)

        # Find values exceeding the threshold
        diff = torch.abs(x[t] - median)  # shape: (D,)
        mask = diff > threshold  # shape: (D,)

        if torch.any(mask):
            has_outlier = True
            corrected_x[t][mask] = median[mask]

    return corrected_x, has_outlier


@torch.jit.script
def _build_single_axis_rot_mat(axis: str, angle: torch.Tensor) -> torch.Tensor:
    """Helper function to build a rotation matrix for a single axis."""
    cos_a = torch.cos(angle)
    sin_a = torch.sin(angle)

    # Create tensors for 0s and 1s with the same batch shape as the angle
    zeros = torch.zeros_like(angle)
    ones = torch.ones_like(angle)

    if axis == "x":
        row1 = torch.stack([ones, zeros, zeros], dim=-1)
        row2 = torch.stack([zeros, cos_a, -sin_a], dim=-1)
        row3 = torch.stack([zeros, sin_a, cos_a], dim=-1)
    elif axis == "y":
        row1 = torch.stack([cos_a, zeros, sin_a], dim=-1)
        row2 = torch.stack([zeros, ones, zeros], dim=-1)
        row3 = torch.stack([-sin_a, zeros, cos_a], dim=-1)
    else:  # axis == 'z'
        row1 = torch.stack([cos_a, -sin_a, zeros], dim=-1)
        row2 = torch.stack([sin_a, cos_a, zeros], dim=-1)
        row3 = torch.stack([zeros, zeros, ones], dim=-1)

    # Stack the rows to form the [..., 3, 3] matrix
    return torch.stack([row1, row2, row3], dim=-2)


def euler_to_rot_mat(axis: str, angle: torch.Tensor) -> torch.Tensor:
    """
    Converts a batch of Euler angles to rotation matrices with variable axis length.

    This function mimics the behavior of `scipy.spatial.transform.Rotation`
    (with the default extrinsic convention), where the length of the `axis`
    string must match the size of the last dimension of the `angle` tensor.

    Args:
        axis (str): A string specifying the axes of rotation, e.g., 'x', 'xy', 'zyx'.
        angle (torch.Tensor): A tensor of Euler angles with shape [..., N],
                              where N is the length of the `axis` string.

    Returns:
        torch.Tensor: The corresponding rotation matrices with shape [..., 3, 3].
    """
    num_axes = len(axis)
    if num_axes == 0:
        raise ValueError("Axis string cannot be empty.")
    if angle.shape[-1] != num_axes:
        raise ValueError(
            f"The number of axes ({num_axes}) must match the "
            f"last dimension of the angle tensor ({angle.shape[-1]})."
        )

    # Get batch shape and device context from the input tensor
    batch_shape = angle.shape[:-1]
    device = angle.device
    dtype = angle.dtype

    # Start with a batch of identity matrices
    total_rot = torch.eye(3, device=device, dtype=dtype).expand(*batch_shape, 3, 3)

    # Sequentially apply rotations
    # For extrinsic rotations, we left-multiply each new rotation matrix.
    # R_total = R_n * ... * R_2 * R_1 * I
    for i in range(num_axes):
        ax = axis[i].lower()
        a = angle[..., i]
        R_i = _build_single_axis_rot_mat(ax, a)
        total_rot = torch.matmul(R_i, total_rot)

    return total_rot


def rot_mat_to_euler(mat: torch.Tensor, axis: str) -> torch.Tensor:
    """
    Converts a batch of rotation matrices to Euler angles for any Tait-Bryan sequence.

    Args:
        mat (torch.Tensor): A tensor of rotation matrices with shape [..., 3, 3].
        axis (str): A 3-character string specifying the desired axis sequence.
                    Supported sequences: 'xyz', 'xzy', 'yxz', 'yzx', 'zxy', 'zyx'.

    Returns:
        torch.Tensor: The corresponding Euler angles with shape [..., 3].
    """
    if mat.shape[-2:] != (3, 3):
        raise ValueError("Input tensor must have shape [..., 3, 3]")

    eps = 1e-8  # Epsilon for clamping and gimbal lock checks
    axis = axis.lower()

    # To avoid confusion, let's name angles based on their order of application
    # angle1 is for axis[0], angle2 for axis[1], angle3 for axis[2]

    if axis == "xyz":
        # angle2(y) = -asin(R31)
        angle2 = -torch.asin(mat[..., 2, 0].clamp(-1 + eps, 1 - eps))
        cos_angle2 = torch.cos(angle2)
        is_gimbal_lock = cos_angle2 < eps

        # Normal case
        angle1_normal = torch.atan2(mat[..., 2, 1], mat[..., 2, 2])
        angle3_normal = torch.atan2(mat[..., 1, 0], mat[..., 0, 0])

        # Gimbal lock case: set angle1(x) = 0
        angle3_gimbal = torch.atan2(-mat[..., 0, 1], mat[..., 1, 1])

        angle1 = torch.where(is_gimbal_lock, torch.zeros_like(angle2), angle1_normal)
        angle3 = torch.where(is_gimbal_lock, angle3_gimbal, angle3_normal)

        return torch.stack([angle1, angle2, angle3], dim=-1)

    elif axis == "xzy":
        # angle2(z) = asin(R21)
        angle2 = torch.asin(mat[..., 1, 0].clamp(-1 + eps, 1 - eps))
        cos_angle2 = torch.cos(angle2)
        is_gimbal_lock = cos_angle2 < eps

        angle1_normal = torch.atan2(-mat[..., 1, 2], mat[..., 1, 1])
        angle3_normal = torch.atan2(-mat[..., 2, 0], mat[..., 0, 0])

        # Gimbal lock: set angle1(x) = 0
        angle3_gimbal = torch.atan2(mat[..., 2, 1], mat[..., 2, 2])

        angle1 = torch.where(is_gimbal_lock, torch.zeros_like(angle2), angle1_normal)
        angle3 = torch.where(is_gimbal_lock, angle3_gimbal, angle3_normal)

        return torch.stack([angle1, angle2, angle3], dim=-1)

    elif axis == "yxz":
        # angle2(x) = asin(R32)
        angle2 = torch.asin(mat[..., 2, 1].clamp(-1 + eps, 1 - eps))
        cos_angle2 = torch.cos(angle2)
        is_gimbal_lock = cos_angle2 < eps

        angle1_normal = torch.atan2(-mat[..., 2, 0], mat[..., 2, 2])
        angle3_normal = torch.atan2(-mat[..., 0, 1], mat[..., 1, 1])

        # Gimbal lock: set angle1(y) = 0
        angle3_gimbal = torch.atan2(mat[..., 0, 2], mat[..., 0, 0])

        angle1 = torch.where(is_gimbal_lock, torch.zeros_like(angle2), angle1_normal)
        angle3 = torch.where(is_gimbal_lock, angle3_gimbal, angle3_normal)

        return torch.stack([angle1, angle2, angle3], dim=-1)

    elif axis == "yzx":
        # angle2(z) = -asin(R12)
        angle2 = -torch.asin(mat[..., 0, 1].clamp(-1 + eps, 1 - eps))
        cos_angle2 = torch.cos(angle2)
        is_gimbal_lock = cos_angle2 < eps

        angle1_normal = torch.atan2(mat[..., 0, 2], mat[..., 0, 0])
        angle3_normal = torch.atan2(mat[..., 2, 1], mat[..., 1, 1])

        # Gimbal lock: set angle1(y) = 0
        angle3_gimbal = torch.atan2(-mat[..., 2, 0], mat[..., 2, 2])

        angle1 = torch.where(is_gimbal_lock, torch.zeros_like(angle2), angle1_normal)
        angle3 = torch.where(is_gimbal_lock, angle3_gimbal, angle3_normal)

        return torch.stack([angle1, angle2, angle3], dim=-1)

    elif axis == "zxy":
        # angle2(x) = asin(R23)
        angle2 = torch.asin(mat[..., 1, 2].clamp(-1 + eps, 1 - eps))
        cos_angle2 = torch.cos(angle2)
        is_gimbal_lock = cos_angle2 < eps

        angle1_normal = torch.atan2(-mat[..., 1, 0], mat[..., 1, 1])
        angle3_normal = torch.atan2(-mat[..., 0, 2], mat[..., 2, 2])

        # Gimbal lock: set angle1(z) = 0
        angle3_gimbal = torch.atan2(mat[..., 0, 1], mat[..., 0, 0])

        angle1 = torch.where(is_gimbal_lock, torch.zeros_like(angle2), angle1_normal)
        angle3 = torch.where(is_gimbal_lock, angle3_gimbal, angle3_normal)

        return torch.stack([angle1, angle2, angle3], dim=-1)

    elif axis == "zyx":
        # angle2(y) = -asin(R31)
        angle2 = -torch.asin(mat[..., 2, 0].clamp(-1 + eps, 1 - eps))
        cos_angle2 = torch.cos(angle2)
        is_gimbal_lock = cos_angle2 < eps

        angle1_normal = torch.atan2(mat[..., 1, 0], mat[..., 0, 0])
        angle3_normal = torch.atan2(mat[..., 2, 1], mat[..., 2, 2])

        # Gimbal lock: set angle3(x) = 0
        angle1_gimbal = torch.atan2(-mat[..., 0, 1], mat[..., 0, 2])

        angle3 = torch.where(is_gimbal_lock, torch.zeros_like(angle2), angle3_normal)
        angle1 = torch.where(is_gimbal_lock, angle1_gimbal, angle1_normal)

        return torch.stack([angle1, angle2, angle3], dim=-1)

    else:
        raise NotImplementedError(f"Axis sequence '{axis}' is not a supported Tait-Bryan sequence.")


@torch.jit.script
def euler_xyz_to_rot_mat(euler_angles: torch.Tensor) -> torch.Tensor:
    """
    Convert Euler angles (roll, pitch, yaw) in 'xyz' order to rotation matrices.

    Parameters:
    - euler_angles (torch.Tensor): Tensor of shape (B, 3) where each row contains [roll, pitch, yaw].

    Returns:
    - rot_mats (torch.Tensor): Rotation matrices of shape (B, 3, 3).
    """
    roll = euler_angles[:, 0]
    pitch = euler_angles[:, 1]
    yaw = euler_angles[:, 2]

    # Compute sin and cos for each angle
    sin_roll = torch.sin(roll)
    cos_roll = torch.cos(roll)
    sin_pitch = torch.sin(pitch)
    cos_pitch = torch.cos(pitch)
    sin_yaw = torch.sin(yaw)
    cos_yaw = torch.cos(yaw)

    # Rotation matrices around the x, y, and z axes
    Rx = torch.stack(
        [
            torch.ones_like(roll),
            torch.zeros_like(roll),
            torch.zeros_like(roll),
            torch.zeros_like(roll),
            cos_roll,
            -sin_roll,
            torch.zeros_like(roll),
            sin_roll,
            cos_roll,
        ],
        dim=1,
    ).reshape(-1, 3, 3)

    Ry = torch.stack(
        [
            cos_pitch,
            torch.zeros_like(pitch),
            sin_pitch,
            torch.zeros_like(pitch),
            torch.ones_like(pitch),
            torch.zeros_like(pitch),
            -sin_pitch,
            torch.zeros_like(pitch),
            cos_pitch,
        ],
        dim=1,
    ).reshape(-1, 3, 3)

    Rz = torch.stack(
        [
            cos_yaw,
            -sin_yaw,
            torch.zeros_like(yaw),
            sin_yaw,
            cos_yaw,
            torch.zeros_like(yaw),
            torch.zeros_like(yaw),
            torch.zeros_like(yaw),
            torch.ones_like(yaw),
        ],
        dim=1,
    ).reshape(-1, 3, 3)

    # Combined rotation matrix: Rz * Ry * Rx
    rot_mats = torch.bmm(Rz, torch.bmm(Ry, Rx))  # (B, 3, 3)

    return rot_mats


@torch.jit.script
def q_mul(q1: torch.Tensor, q2: torch.Tensor) -> torch.Tensor:
    """
    Multiply two quaternions, q1 and q2.
    Expects tensors of shape (..., 4).
    """
    # Extract components
    w1, x1, y1, z1 = q1[..., 0], q1[..., 1], q1[..., 2], q1[..., 3]
    w2, x2, y2, z2 = q2[..., 0], q2[..., 1], q2[..., 2], q2[..., 3]

    # Quaternion multiplication formula
    w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
    z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2

    # Stack components back into a tensor
    return torch.stack((w, x, y, z), dim=-1)


def euler_to_quaternion(e: torch.Tensor, order: str) -> torch.Tensor:
    """
    Convert Euler angles to quaternions using PyTorch.

    Args:
        e (torch.Tensor): Euler angles, tensor of shape (..., 3).
        order (str): The order of rotations (e.g., 'xyz', 'zyx').

    Returns:
        torch.Tensor: Quaternions, tensor of shape (..., 4).
    """
    assert e.shape[-1] == 3

    original_shape = list(e.shape)
    original_shape[-1] = 4

    # Reshape to a 2D tensor for easier processing
    e = e.reshape(-1, 3)

    x = e[:, 0]
    y = e[:, 1]
    z = e[:, 2]

    # Individual rotation quaternions
    rx = torch.stack((torch.cos(x / 2), torch.sin(x / 2), torch.zeros_like(x), torch.zeros_like(x)), dim=1)
    ry = torch.stack((torch.cos(y / 2), torch.zeros_like(y), torch.sin(y / 2), torch.zeros_like(y)), dim=1)
    rz = torch.stack((torch.cos(z / 2), torch.zeros_like(z), torch.zeros_like(z), torch.sin(z / 2)), dim=1)

    result = None
    for coord in order:
        if coord == "x":
            r = rx
        elif coord == "y":
            r = ry
        elif coord == "z":
            r = rz
        else:
            raise ValueError(f"Unknown coordinate '{coord}' in order string")

        if result is None:
            result = r
        else:
            result = q_mul(result, r)

    # Reverse antipodal representation for specific orders to ensure a non-negative "w" component,
    # which is a common convention.
    if result is not None and order in ["xyz", "yzx", "zxy"]:
        result = result * -1  # Use multiplication instead of in-place operation

    return result.reshape(original_shape) if result is not None else None


@torch.jit.script
def quat_fix(q):
    """
    Enforce quaternion continuity across the time dimension by selecting
    the representation (q or -q) with minimal distance (or, equivalently, maximal dot product)
    between two consecutive frames.

    Expects a tensor of shape [..., T, 4], where T is the sequence length (time dimension).
    The time dimension is assumed to be the second-to-last dimension.
    Returns a tensor of the same shape.

    Args:
        q: torch.Tensor of shape [..., T, 4] containing quaternions

    Returns:
        torch.Tensor of the same shape with quaternion continuity enforced
    """
    assert q.shape[-1] == 4, f"Last dimension must be 4 for quaternions, got {q.shape[-1]}"
    assert q.dim() >= 2, f"Input must have at least 2 dimensions, got {q.dim()}"

    # Get the time dimension (second-to-last)
    time_dim = q.shape[-2]
    assert time_dim >= 2, f"Time dimension must be at least 2, got {time_dim}"

    # Clone the input to avoid modifying the original tensor
    result = q.clone()

    # Calculate dot products between consecutive frames
    # q[..., 1:, :] * q[..., :-1, :] -> [..., T-1, 4]
    # Sum over the last dimension (quaternion components) -> [..., T-1]
    dot_products = torch.sum(q[..., 1:, :] * q[..., :-1, :], dim=-1)

    # Create mask for frames where dot product is negative
    mask = dot_products < 0.0

    # Apply cumulative sum along time dimension and check for odd counts
    # This determines which frames need to be flipped
    cumsum_mask = torch.cumsum(mask, dim=-1) % 2
    flip_mask = cumsum_mask.to(torch.bool)

    # Apply the flipping to the result
    # We need to flip q[..., 1:, :] where flip_mask is True
    result[..., 1:, :][flip_mask] *= -1

    return result


@torch.jit.script
def rot_vec_to_mat(rot_vec: torch.Tensor) -> torch.Tensor:
    """
    Convert rotation vectors to rotation matrices using Rodrigues' formula.

    Args:
        rot_vec (torch.Tensor): Rotation vectors of shape (B, 3).

    Returns:
        torch.Tensor: Rotation matrices of shape (B, 3, 3).
    """
    # Ensure input is of shape (B, 3)
    assert rot_vec.dim() == 2 and rot_vec.size(1) == 3, "Input must be of shape (B, 3)"

    B = rot_vec.size(0)

    # Compute the angle (theta) for each rotation vector
    theta = torch.norm(rot_vec, dim=1, keepdim=True)  # Shape: (B, 1)

    # To avoid division by zero, set theta to 1 where theta is zero temporarily
    # We'll handle theta=0 cases separately
    theta_nonzero = theta.clone()
    theta_nonzero[theta_nonzero == 0] = 1.0

    # Compute the unit rotation axis
    k = rot_vec / theta_nonzero  # Shape: (B, 3)

    # Skew-symmetric matrices for each k
    K = torch.zeros((B, 3, 3), dtype=rot_vec.dtype, device=rot_vec.device)
    K[:, 0, 1] = -k[:, 2]
    K[:, 0, 2] = k[:, 1]
    K[:, 1, 0] = k[:, 2]
    K[:, 1, 2] = -k[:, 0]
    K[:, 2, 0] = -k[:, 1]
    K[:, 2, 1] = k[:, 0]

    # Compute sine and cosine of theta
    sin_theta = torch.sin(theta)  # Shape: (B, 1)
    cos_theta = torch.cos(theta)  # Shape: (B, 1)

    # Compute K squared
    K_squared = torch.bmm(K, K)  # Shape: (B, 3, 3)

    # Compute the rotation matrix using Rodrigues' formula
    # R = I + sin(theta) * K + (1 - cos(theta)) * K^2
    identity_matrix = torch.eye(3, dtype=rot_vec.dtype, device=rot_vec.device).unsqueeze(0).repeat(B, 1, 1)
    R = identity_matrix + sin_theta.view(B, 1, 1) * K + (1 - cos_theta).view(B, 1, 1) * K_squared

    # Handle the case when theta is very small by using the first order approximation
    # R ≈ I + K for small theta
    small_theta = (theta < 1e-6).squeeze(1)  # Shape: (B,)
    if small_theta.any():
        # For small angles, use Taylor expansion: sin(theta) ≈ theta, (1 - cos(theta)) ≈ theta^2 / 2
        theta_sq = theta[small_theta] ** 2
        sin_theta_small = theta[small_theta]
        one_minus_cos_theta_small = theta_sq / 2

        K_small = K[small_theta]
        K_squared_small = K_squared[small_theta]

        R_small = (
            identity_matrix[small_theta]
            + sin_theta_small.view(-1, 1, 1) * K_small
            + one_minus_cos_theta_small.view(-1, 1, 1) * K_squared_small
        )
        # Alternatively, for very small theta, R ≈ I + K
        # R_small = I[small_theta] + K_small
        R[small_theta] = R_small

    return R


@torch.jit.script
def rot_mat_to_vec(rot_mat: torch.Tensor) -> torch.Tensor:
    """
    Convert rotation matrices to rotation vectors (axis-angle representation).

    Args:
        rot_mat (torch.Tensor): Rotation matrices of shape (B, 3, 3).

    Returns:
        torch.Tensor: Rotation vectors of shape (B, 3).
    """
    # Ensure input is of shape (B, 3, 3)
    assert rot_mat.dim() == 3 and rot_mat.size(1) == 3 and rot_mat.size(2) == 3, "Input must be of shape (B, 3, 3)"

    B = rot_mat.size(0)

    # Compute the trace of each rotation matrix
    trace = rot_mat[:, 0, 0] + rot_mat[:, 1, 1] + rot_mat[:, 2, 2]  # Shape: (B,)

    # Compute the rotation angle theta
    # Clamp the trace to avoid numerical errors outside the valid range for arccos
    trace_clamped = torch.clamp((trace - 1) / 2, -1.0, 1.0)
    theta = torch.acos(trace_clamped)  # Shape: (B,)

    # Compute the sin(theta) for later use
    sin_theta = torch.sin(theta)  # Shape: (B,)

    # Initialize rotation vectors
    rot_vec = torch.zeros((B, 3), dtype=rot_mat.dtype, device=rot_mat.device)

    # Identify cases where theta is not close to 0 or pi
    # Using a small epsilon to handle numerical stability
    epsilon = 1e-6
    not_zero = (theta > epsilon) & (theta < (torch.pi - epsilon))  # Shape: (B,)

    # For general cases where theta is not 0 or pi
    if not_zero.any():
        # Extract the skew-symmetric part of the rotation matrix
        K = torch.zeros((B, 3, 3), dtype=rot_mat.dtype, device=rot_mat.device)
        K[:, 0, 1] = rot_mat[:, 2, 0] - rot_mat[:, 0, 2]
        K[:, 0, 2] = rot_mat[:, 1, 0] - rot_mat[:, 0, 1]
        K[:, 1, 2] = rot_mat[:, 0, 1] - rot_mat[:, 1, 0]
        # Since K is skew-symmetric, K[:,1,0] = -K[:,0,1], etc., but we only need the upper triangle

        # Compute the rotation axis
        # axis = (1 / (2 * sin(theta))) * [R32 - R23, R13 - R31, R21 - R12]
        axis = torch.zeros((B, 3), dtype=rot_mat.dtype, device=rot_mat.device)
        axis[not_zero, 0] = rot_mat[not_zero, 2, 1] - rot_mat[not_zero, 1, 2]
        axis[not_zero, 1] = rot_mat[not_zero, 0, 2] - rot_mat[not_zero, 2, 0]
        axis[not_zero, 2] = rot_mat[not_zero, 1, 0] - rot_mat[not_zero, 0, 1]
        axis[not_zero] = axis[not_zero] / (2 * sin_theta[not_zero].unsqueeze(1))

        # Compute the rotation vectors
        rot_vec[not_zero] = axis[not_zero] * theta[not_zero].unsqueeze(1)

    # Handle the zero rotation case: rot_vec remains zero
    # Handle the 180-degree rotation case
    # When theta is pi, special handling is required to extract the rotation axis
    is_pi = torch.abs(theta - torch.pi) < epsilon  # Shape: (B,)
    if is_pi.any():
        # For theta == pi, the rotation axis can be extracted from the diagonal elements
        # of the rotation matrix. Specifically, (Rii + 1)/2 = k_i^2
        # To avoid numerical issues, choose the largest diagonal element
        R = rot_mat[is_pi]  # Shape: (Np, 3, 3)
        rot_vec_pi = torch.zeros((R.size(0), 3), dtype=rot_mat.dtype, device=rot_mat.device)

        # Compute (R + I) / 2
        R_plus_I = (R + torch.eye(3, dtype=rot_mat.dtype, device=rot_mat.device).unsqueeze(0)) / 2

        # Find the axis with the largest diagonal element
        diag = torch.diagonal(R_plus_I, dim1=1, dim2=2)  # Shape: (Np, 3)
        max_diag, max_idx = diag.max(dim=1)  # Shape: (Np,)

        # Initialize axis
        axis = torch.zeros_like(R_plus_I)
        indices = torch.arange(R.size(0), device=rot_mat.device, dtype=torch.long)
        axis[indices, max_idx, max_idx] = 1.0

        # For each rotation matrix where theta == pi
        for i in range(R.size(0)):
            idx = max_idx[i]
            axis_i = torch.zeros(3, dtype=rot_mat.dtype, device=rot_mat.device)
            axis_i[idx] = torch.sqrt(R_plus_I[i, idx, idx] + 1e-8)  # Add small epsilon to avoid sqrt(0)

            # Determine the signs of the other two components
            j = (idx + 1) % 3
            k = (idx + 2) % 3
            if R[i, j, k] >= 0:
                axis_i[j] = R[i, j, k] / (2 * axis_i[idx] + 1e-8)
            else:
                axis_i[j] = -R[i, j, k] / (2 * axis_i[idx] + 1e-8)

            if R[i, k, j] >= 0:
                axis_i[k] = R[i, k, j] / (2 * axis_i[idx] + 1e-8)
            else:
                axis_i[k] = -R[i, k, j] / (2 * axis_i[idx] + 1e-8)

            # Normalize the axis to ensure it's a unit vector
            axis_i = axis_i / (torch.norm(axis_i) + 1e-8)

            # Assign to rotation vector
            rot_vec_pi[i] = axis_i * torch.pi

        rot_vec[is_pi] = rot_vec_pi

    return rot_vec


@torch.jit.script
def batch_rot_vec_to_mat(rot_vec: torch.Tensor) -> torch.Tensor:
    # input: (B, N, 3)
    # output: (B, N, 3, 3)
    B, N, _ = rot_vec.shape
    rot_mat = rot_vec_to_mat(rot_vec.reshape(-1, 3)).reshape(B, N, 3, 3)
    return rot_mat


@torch.jit.script
def batch_rot_mat_to_vec(rot_mat: torch.Tensor) -> torch.Tensor:
    # input: (B, N, 3, 3)
    # output: (B, N, 3)
    B, N, _, _ = rot_mat.shape
    rot_vec = rot_mat_to_vec(rot_mat.reshape(-1, 3, 3)).reshape(B, N, 3)
    return rot_vec


@torch.jit.script
def angle_axis_to_ypr(angle_axis: torch.Tensor) -> torch.Tensor:
    """
    Convert angle-axis representation to yaw-pitch-roll angles.

    Args:
        angle_axis (torch.Tensor): Angle-axis representation of shape (B, 3).

    Returns:
        torch.Tensor: Yaw, pitch, and roll angles.
    """
    # Ensure input is of shape (B, 3)
    assert angle_axis.dim() == 2 and angle_axis.size(1) == 3, "Input must be of shape (B, 3)"

    rot_mat = rot_vec_to_mat(angle_axis)  # Convert to rotation matrices
    yaw = torch.atan2(rot_mat[:, 0, 1], rot_mat[:, 0, 0])  # Yaw angle
    pitch = torch.atan2(-rot_mat[:, 0, 2], torch.sqrt(rot_mat[:, 1, 2] ** 2 + rot_mat[:, 2, 2] ** 2))  # Pitch angle
    roll = torch.atan2(rot_mat[:, 1, 2], rot_mat[:, 2, 2])  # Roll angle
    return torch.stack([yaw, pitch, roll], dim=1)  # Shape: (batch_size, 3)


@torch.jit.script
def batch_angle_axis_to_ypr(angle_axis: torch.Tensor) -> torch.Tensor:
    B, N, _ = angle_axis.shape
    ypr = angle_axis_to_ypr(angle_axis.reshape(-1, 3))
    return ypr.reshape(B, N, 3)


def swap_order(x: torch.Tensor, swap_pairs: List[List[int]], axis: int) -> torch.Tensor:
    """
    Swap elements in a tensor based on specified swap pairs along a given axis.

    Args:
        x (torch.Tensor): Input tensor.
        swap_pairs (List[List[int]]): List of pairs indicating indices to swap.
                                      Each pair is a list or tuple of two integers.
        axis (int): The axis along which to perform the swaps.

    Returns:
        torch.Tensor: Tensor with specified elements swapped.
    """
    # Convert swap_pairs to a Torch tensor
    swap_pairs_tensor = torch.tensor(swap_pairs, dtype=torch.long, device=x.device)

    # Separate the indices to swap
    a_indices = swap_pairs_tensor[:, 0]
    b_indices = swap_pairs_tensor[:, 1]

    # Prepare indexing for axis
    # Create a list of slice(None) for all dimensions
    idx_a = [slice(None)] * x.dim()
    idx_b = [slice(None)] * x.dim()

    # Assign the swap indices to the specified axis
    idx_a[axis] = a_indices
    idx_b[axis] = b_indices

    # Clone the elements at a_indices to avoid in-place overwriting
    temp = x[tuple(idx_a)].clone()

    # Perform the swaps
    x_swapped = x.clone()
    x_swapped[tuple(idx_a)] = x[tuple(idx_b)]
    x_swapped[tuple(idx_b)] = temp

    return x_swapped


def symmetry_smplh_pose(pose: torch.Tensor) -> torch.Tensor:
    """
    Symmetrize SMPLH pose by swapping specified joint rotations.

    Args:
        pose (torch.Tensor): Input tensor of shape (B, N), where B is batch size,
                             N is the number of samples, and 63 corresponds to 21 joints × 3 coordinates.

    Returns:
        torch.Tensor: Symmetrized pose tensor of shape (B, N).
    """
    assert pose.shape[-1] == 63, "Input should be SMPL pose with 21 joints"

    # Define joint pairs to swap (1-based indexing as per original code)
    swap_pairs = [[1, 2], [4, 5], [7, 8], [10, 11], [13, 14], [16, 17], [18, 19], [20, 21]]
    swap_pairs = [[x - 1 for x in pair] for pair in swap_pairs]  # Convert to zero-based indexing

    # # Convert to zero-based indexing and create a Torch tensor
    # swap_index = torch.tensor(swap_pairs, dtype=torch.long, device=pose.device) - 1  # Shape: (8, 2)

    B, N = pose.shape[:2]
    pose = pose.reshape(-1, 21, 3)  # Shape: (B * N, 21, 3)

    # Convert rotation vectors to rotation matrices
    mat_opp = batch_rot_vec_to_mat(pose)  # Shape: (B * N, 21, 3, 3)

    # Negate specific elements for symmetry
    mat_opp[:, :, 0, 1] *= -1
    mat_opp[:, :, 0, 2] *= -1
    mat_opp[:, :, 1, 0] *= -1
    mat_opp[:, :, 2, 0] *= -1

    # Swap specific joints based on the provided swap pairs
    mat_opp = swap_order(mat_opp, swap_pairs, axis=1)

    # Convert rotation matrices back to rotation vectors
    pose_opp = batch_rot_mat_to_vec(mat_opp).reshape(B, -1)  # Shape: (B, N)

    return pose_opp


@torch.jit.script
def angle_axis_to_quaternion(angle_axis: torch.Tensor) -> torch.Tensor:
    """
    Converts angle-axis representation to quaternion.
    angle_axis: Tensor of shape (..., 3) where each row is an axis-angle vector.
    """
    angle = torch.norm(angle_axis, dim=-1, keepdim=True)  # Compute the angle
    small_angle_mask = angle < 1e-6  # Threshold to handle small angles

    # Normalize axis and avoid division by zero
    axis = angle_axis / (angle + 1e-6)

    # Compute quaternion components
    qw = torch.cos(angle / 2)
    sin_half_angle = torch.sin(angle / 2)
    qx = axis[..., 0:1] * sin_half_angle
    qy = axis[..., 1:2] * sin_half_angle
    qz = axis[..., 2:3] * sin_half_angle

    # Handle small angles: output identity quaternion (no rotation)
    qw = torch.where(small_angle_mask, torch.ones_like(qw), qw)
    qx = torch.where(small_angle_mask, torch.zeros_like(qx), qx)
    qy = torch.where(small_angle_mask, torch.zeros_like(qy), qy)
    qz = torch.where(small_angle_mask, torch.zeros_like(qz), qz)

    # Combine into a quaternion (..., 4)
    return torch.cat([qw, qx, qy, qz], dim=-1)


# SLERP Implementation in PyTorch
@torch.jit.script
def slerp(q1, q2, t):
    """
    Perform spherical linear interpolation (SLERP) between two quaternions.
    Args:
        q1, q2: Tensors of shape (..., 4) - input quaternions.
        t: Interpolation factor (scalar or tensor in [0, 1]).
    """
    # Ensure quaternions are normalized
    q1 = q1 / (torch.norm(q1, dim=-1, keepdim=True) + 1e-6)
    q2 = q2 / (torch.norm(q2, dim=-1, keepdim=True) + 1e-6)

    # Compute dot product and clamp it
    dot_product = torch.sum(q1 * q2, dim=-1, keepdim=True)
    dot_product = torch.clamp(dot_product, -1.0, 1.0)  # Avoid acos out of range

    # Calculate the angle between quaternions
    theta_0 = torch.acos(dot_product)
    sin_theta_0 = torch.sin(theta_0)

    # Handle case when sin(theta_0) is close to zero
    small_angle_mask = sin_theta_0 < 1e-6
    s0 = torch.where(small_angle_mask, 1.0 - t, torch.sin((1 - t) * theta_0) / (sin_theta_0 + 1e-6))
    s1 = torch.where(small_angle_mask, t, torch.sin(t * theta_0) / (sin_theta_0 + 1e-6))

    return s0 * q1 + s1 * q2


# Convert Quaternion to Angle-Axis
@torch.jit.script
def quaternion_to_angle_axis(quaternion: torch.Tensor) -> torch.Tensor:
    qw, qx, qy, qz = quaternion[..., 0], quaternion[..., 1], quaternion[..., 2], quaternion[..., 3]
    angle = 2 * torch.acos(torch.clamp(qw, -1.0, 1.0))  # Avoid numerical instability
    s = torch.sqrt(1 - qw**2 + 1e-6)  # To avoid division by zero
    axis = torch.stack([qx, qy, qz], dim=-1) / s.unsqueeze(-1)
    return axis * angle.unsqueeze(-1)


def downsample_angle_axis(angle_axis_data: torch.Tensor, target_fps: int = 50, source_fps: int = 120) -> torch.Tensor:
    """
    Downsamples angle-axis MoCap data from source_fps to target_fps.
    Supports multiple joints (C = num_joints * 3).

    Args:
        angle_axis_data (torch.Tensor): Tensor of shape (num_frames, C) where C = num_joints * 3
        target_fps (int): Target FPS (must be less than source_fps, e.g., 50)
        source_fps (int): Source FPS (e.g., 120)

    Returns:
        torch.Tensor: Downsampled angle-axis data of shape (new_num_frames, C).
    """
    assert target_fps < source_fps, "target_fps must be less than source_fps"
    assert angle_axis_data.dim() == 2 and angle_axis_data.size(1) % 3 == 0, "Invalid angle_axis_data shape"

    num_frames, C = angle_axis_data.shape
    num_joints = C // 3

    # Convert angle-axis to quaternions: shape (num_frames, num_joints, 4)
    quaternions = angle_axis_to_quaternion(angle_axis_data.reshape(-1, 3))  # (num_frames * num_joints, 4)
    quaternions = quaternions.reshape(num_frames, num_joints, 4)

    # Calculate the total duration
    duration = num_frames / source_fps  # in seconds

    # Determine the number of frames after downsampling
    new_num_frames = int(np.ceil(duration * target_fps))
    if new_num_frames < 1:
        raise ValueError("Resulting number of frames is less than 1. Choose a higher target_fps or shorter duration.")

    # Generate target time indices
    t_target = torch.linspace(0, duration, steps=new_num_frames, device=angle_axis_data.device)

    # Perform SLERP interpolation for each target time
    target_quaternions = []
    for t in t_target:
        # Find the corresponding position in the source frames
        exact_frame = t * source_fps
        idx = torch.floor(exact_frame).long()
        t_ratio = exact_frame - idx.float()

        # Clamp indices to avoid out-of-bounds
        idx = torch.clamp(idx, max=num_frames - 2)

        q1 = quaternions[idx]  # Shape: (num_joints, 4)
        q2 = quaternions[idx + 1]  # Shape: (num_joints, 4)

        # Perform SLERP between q1 and q2
        q_interpolated = slerp(q1, q2, t_ratio)
        target_quaternions.append(q_interpolated)

    # Stack interpolated quaternions: (new_num_frames, num_joints, 4)
    target_quaternions = torch.stack(target_quaternions, dim=0)

    # Convert quaternions back to angle-axis: (new_num_frames, num_joints, 3)
    downsampled_angle_axis = quaternion_to_angle_axis(target_quaternions.reshape(-1, 4)).reshape(new_num_frames, C)

    return downsampled_angle_axis


# Interpolation Pipeline
def interpolate_angle_axis(
    angle_axis_data: torch.Tensor, target_fps: int = 1000, source_fps: torch.Tensor = 120
) -> torch.Tensor:
    """
    Interpolates angle-axis MoCap data from source_fps to target_fps.
    Supports multiple joints (C = num_joints * 3).

    Args:
        angle_axis_data: Tensor of shape (num_frames, C) where C = num_joints * 3
        target_fps: Target FPS (e.g., 1000)
        source_fps: Source FPS (e.g., 120)

    Returns:
        Interpolated angle-axis data of shape (new_num_frames, C).
    """
    if target_fps == source_fps:
        return angle_axis_data
    elif target_fps < source_fps:
        return downsample_angle_axis(angle_axis_data, target_fps, source_fps)
    num_frames, C = angle_axis_data.shape
    num_joints = C // 3  # Number of joints

    # Convert to quaternions: shape (num_frames, num_joints, 4)
    quaternions = angle_axis_to_quaternion(angle_axis_data.reshape(-1, 3))
    quaternions = quaternions.reshape(num_frames, num_joints, 4)
    # Create time indices
    duration = num_frames / source_fps  # Total duration in seconds
    t_original = torch.linspace(0, duration, num_frames, device=angle_axis_data.device)
    t_target = torch.linspace(0, duration, int(num_frames * target_fps / source_fps), device=angle_axis_data.device)

    # SLERP Interpolation
    target_quaternions = []
    for i in range(len(t_target) - 1):
        t = (t_target[i] % (t_original[1] - t_original[0])) / (
            t_original[1] - t_original[0]
        )  # Normalize interpolation ratio
        idx = (t_target[i] // (t_original[1] - t_original[0])).long().clamp(max=num_frames - 2)  # Frame index
        q1 = quaternions[idx]  # Shape: (num_joints, 4)
        q2 = quaternions[idx + 1]  # Shape: (num_joints, 4)
        q_interpolated = slerp(q1, q2, t)
        target_quaternions.append(q_interpolated)

    # Stack all interpolated quaternions: (new_num_frames, num_joints, 4)
    target_quaternions = torch.stack(target_quaternions, dim=0)

    # Convert back to angle-axis: (new_num_frames, num_joints, 3)
    interpolated_angle_axis = quaternion_to_angle_axis(target_quaternions.reshape(-1, 4)).reshape(-1, num_joints * 3)
    return interpolated_angle_axis


def interpolate_trans(trans: torch.Tensor, target_fps: int = 1000, source_fps: torch.Tensor = 120) -> torch.Tensor:
    """
    Interpolates translation data from source_fps to target_fps.

    Args:
        trans: Tensor of shape (num_frames, any)
        target_fps: Target FPS (e.g., 1000)
        source_fps: Source FPS (e.g., 120)

    Returns:
        Interpolated translation of shape (new_num_frames, any).
    """

    num_frames, _ = trans.shape
    if num_frames <= 2:
        return trans

    duration = num_frames / source_fps  # Total duration in seconds
    t_original = torch.linspace(0, duration, num_frames, device=trans.device)
    t_target = torch.linspace(0, duration, int(num_frames * target_fps / source_fps), device=trans.device)

    target_trans = []
    for i in range(len(t_target)):
        t = (t_target[i] % (t_original[1] - t_original[0])) / (
            t_original[1] - t_original[0]
        )  # Normalize interpolation ratio
        idx = (t_target[i] // (t_original[1] - t_original[0])).long().clamp(max=num_frames - 2)  # Frame index
        trans1 = trans[idx]  # Shape: (num_joints, 4)
        trans2 = trans[idx + 1]  # Shape: (num_joints, 4)
        trans_interpolated = trans1 * (1 - t) + trans2 * t
        target_trans.append(trans_interpolated)

    # Stack all interpolated translations: (new_num_frames, 3)
    interpolated_trans = torch.stack(target_trans, dim=0)

    return interpolated_trans


def resample_linear(data: torch.Tensor, target_fps: int, source_fps: int) -> torch.Tensor:
    """
    Resamples data from source_fps to target_fps using linear interpolation.
    This function handles both downsampling and upsampling (interpolation).
    
    Args:
        trans: Tensor of shape (num_frames, any)
        target_fps: Target FPS (e.g., 1000)
        source_fps: Source FPS (e.g., 120)

    Returns:
        Interpolated translation of shape (new_num_frames, any).
        
    """
    if data.shape[0] <= 1:
        return data.clone()
        
    if source_fps == target_fps:
        return data.clone()
    
    num_frames = data.shape[0]
    device = data.device
    dtype = data.dtype

    duration = (num_frames - 1) / source_fps

    target_frames = int(duration * target_fps) + 1

    target_times = torch.linspace(0, duration, target_frames, device=device, dtype=dtype)

    source_indices_float = target_times * source_fps
    
    left_indices = torch.floor(source_indices_float).long()
    weights = source_indices_float - left_indices.float()
    is_last_frame = (source_indices_float >= (num_frames - 1))
    
    left_indices = torch.clamp(left_indices, 0, num_frames - 2)
    right_indices = left_indices + 1
    
    weights_shape = [target_frames] + [1] * (len(data.shape) - 1)
    weights = weights.view(weights_shape)
    is_last_frame = is_last_frame.view(weights_shape)
    
    result = data[left_indices] * (1 - weights) + data[right_indices] * weights

    result = torch.where(is_last_frame, data[-1], result)
    
    return result

def resample_quaternions(
    quaternions: torch.Tensor, target_fps: int, source_fps: int
) -> torch.Tensor:
    """
    Resamples quaternion data from source_fps to target_fps using vectorized slerp.
    This function handles both downsampling and upsampling (interpolation).

    Args:
        quaternions (torch.Tensor): Tensor of shape (num_frames, C) where C = num_joints * 4.
        target_fps (int): Target frames per second.
        source_fps (int): Source frames per second.

    Returns:
        torch.Tensor: Resampled quaternions of shape (new_num_frames, C).
    """
    num_frames, C = quaternions.shape
    assert C % 4 == 0, "Input tensor C dimension must be a multiple of 4"
    num_joints = C // 4

    if target_fps == source_fps or num_frames < 2:
        return quaternions.clone()

    # Calculate total duration and new number of frames
    duration = (num_frames - 1) / source_fps
    new_num_frames = int(duration * target_fps) + 1
    
    # Generate target time indices
    t_target = torch.linspace(0, duration, steps=new_num_frames, device=quaternions.device)
    exact_frames = t_target * source_fps

    # Find the indices of the two source frames to interpolate between
    idx0 = torch.floor(exact_frames).long()
    idx0 = torch.clamp(idx0, 0, num_frames - 2)
    idx1 = idx0 + 1

    # Calculate the interpolation ratio (alpha) for each target frame
    t_ratio = exact_frames - idx0.float()
    t_ratio = t_ratio.reshape(-1, 1, 1)

    # Get the quaternion pairs for interpolation
    q1 = quaternions[idx0]
    q2 = quaternions[idx1]

    # Reshape for slerp
    q1_reshaped = q1.reshape(new_num_frames, num_joints, 4)
    q2_reshaped = q2.reshape(new_num_frames, num_joints, 4)
    
    # Perform vectorized slerp
    q_interpolated = slerp(q1_reshaped, q2_reshaped, t_ratio)

    # Reshape back to the original layout
    result = q_interpolated.reshape(new_num_frames, C)
    
    result[0] = quaternions[0].clone()
    result[-1] = quaternions[-1].clone()

    result_reshaped = result.reshape(-1, 4)
    norms = torch.norm(result_reshaped, dim=1, keepdim=True)
    result_reshaped = result_reshaped / (norms + 1e-8)
    result = result_reshaped.reshape(new_num_frames, C)

    result[0] = quaternions[0].clone()
    result[-1] = quaternions[-1].clone()

    return result

def find_longest_cyclic_subsequence(
    input_tensor: torch.Tensor, max_distance: float = np.inf
) -> Tuple[int, int, int, int]:
    """
    Find the longest cyclic subsequence where the start and end points have the minimum distance.

    Args:
        input_tensor (torch.Tensor): Input tensor of shape (N, 29).
        max_distance (float): Maximum allowable distance between the start and end points (default is no limit).

    Returns:
        tuple: (best_start, best_end, max_length, seq_distance), which are the start index, end index, and the subsequence length, sequence distance.
    """
    # Validate the input tensor's shape
    if input_tensor.dim() != 2:
        raise ValueError("The input tensor must be a 2D tensor with shape (N, num_dof).")

    # Convert the tensor to a NumPy array
    reduced_array = input_tensor.cpu().numpy()

    # Build the KDTree
    tree = KDTree(reduced_array)

    N = reduced_array.shape[0]
    best_start, best_end = 0, 0
    max_length = 0

    # Iterate through each starting point
    for start in range(N):
        # Find points within max_distance
        nearby_indices = tree.query_ball_point(reduced_array[start], max_distance)

        # Iterate through the candidate end points and compute lengths
        for end in nearby_indices:
            if end > start:  # Ensure the subsequence is valid and continuous
                length = end - start
                if length > max_length:
                    max_length = length
                    best_start, best_end = start, end

    seq_distance = torch.norm(input_tensor[best_start] - input_tensor[best_end], dim=-1).item()
    return best_start, best_end, max_length, seq_distance


def contact_to_phase(contact: torch.Tensor, threshold: float = 0.5) -> torch.Tensor:
    """
    Convert a 0/1 contact sequence into sin/cos phase channels.

    Args:
        contact : 1-D torch.Tensor (int/bool) of shape (T,). 0 = swing, 1 = stance

    Returns:
        alpha : 1-D torch.Tensor of shape (T,), where sgn(sin(alpha * 2 * pi)) = contact

    """
    assert contact.dim() == 1, "input must be 1-D"
    assert 0.0 < threshold < 1.0, "threshold must be between 0 and 1"

    # Detect state-change positions
    diff = contact[1:] != contact[:-1]
    edges = torch.nonzero(diff, as_tuple=False).flatten() + 1
    edges = torch.cat([contact.new_tensor([0]), edges, contact.new_tensor([contact.size(0)])])
    edges = edges.long()

    alpha = torch.zeros_like(contact, dtype=torch.float32)

    # Piece-wise linear ramp according to contact value
    for start, end in zip(edges[:-1], edges[1:]):
        length = int(end - start)
        if length == 0:
            continue
        if contact[start] == 1:  # stance
            alpha[start:end] = torch.linspace(0.0, threshold, length, device=contact.device, dtype=torch.float32)
        else:  # swing
            alpha[start:end] = torch.linspace(threshold, 1.0, length, device=contact.device, dtype=torch.float32)

    return alpha


def filt_feet_contact(
    actions: torch.Tensor,
    root_pos: torch.Tensor,
    root_rot: torch.Tensor,
    fk: object,
    foot_names: List[str],
    threshold: float = 0.1,
    foot_ground_offset=0.045,
    disable_height: bool = True,
    debug_visualize: bool = False,
    debug_save_dir: str | None = None,
    debug_rv: object = None,
) -> torch.Tensor:
    """
    Generate feet contact singals (0 or 1) based on the distance between the feet and the ground.

    Args:
        actions (torch.Tensor): Action sequence of shape (B, N, num_dof). N is the sequence length. We assume that for the whole sequence, the foot must be on the ground for at least one frame.
        root_pos (torch.Tensor): Root position of shape (B, N, 3).
        fk (RobotKinematics): RobotKinematics model
        foot_names (List[str]): Names for corresponding links
        threshold (float, optional): Filter param for feet contact. Default is 0.1.
        disable_height (bool, optional): If True, disable height filtering. Default is False.

    Returns:
        feet_contact_indices (torch.Tensor)
        filtered_root_pos (torch.Tensor)
    """
    fk.set_target_links(foot_names)
    feet_contact_indices = []
    filtered_root_poses = []
    for i in range(actions.shape[0]):
        link_pos = fk(actions[i], root_trans_offset=root_pos[i], root_rot=root_rot[i])  # shape (N, 2, 3)
        link_vel = torch.concatenate([torch.zeros_like(link_pos[:1, :, :]), torch.diff(link_pos, dim=0)], dim=0)
        link_vel = torch.linalg.norm(link_vel, dim=-1)  # shape (N, 2)
        link_height = link_pos[:, :, 2]  # shape (N, 2)
        min_link_height, min_link_height_indices = torch.min(link_height, dim=0)  # shape (2,)
        min_link_height, idx = torch.min(min_link_height, dim=0)  # shape (1,)
        min_link_height_indices = min_link_height_indices[idx]  # shape (1,)
        # Calculate the height difference between the root and the lowest point
        # height_diff = root_pos[i][min_link_height_indices, 2] - min_link_height + foot_ground_offset
        cond_0 = link_pos[:, 0, 2] < min_link_height + threshold * (
            root_pos[i][min_link_height_indices, 2] - min_link_height
        )  # Condition for link_pos[:, 0, 2]

        cond_1 = link_pos[:, 1, 2] < min_link_height + threshold * (
            root_pos[i][min_link_height_indices, 2] - min_link_height
        )  # Condition for link_pos[:, 1, 2]

        # Initialize the result tensor with -1 (or any other placeholder)
        height_filtered_indices = torch.full((link_pos.shape[0], 2), -1, dtype=torch.long).to(link_pos.device)

        # Set the row indices where the condition is True for each column
        height_filtered_indices[cond_0, 0] = torch.arange(link_pos.shape[0]).to(link_pos.device)[
            cond_0
        ]  # row indices for column 0
        height_filtered_indices[cond_1, 1] = torch.arange(link_pos.shape[0]).to(link_pos.device)[
            cond_1
        ]  # row indices for column 1

        # Create conditions to filter out velocities below a certain threshold
        cond_vel_0 = link_vel[:, 0] <= 0.6 * threshold
        cond_vel_1 = link_vel[:, 1] <= 0.6 * threshold

        # Initialize the result tensor for velocities with -1 (or any other placeholder)
        vel_filtered_indices = torch.full((link_vel.shape[0], 2), -1, dtype=torch.long).to(link_vel.device)

        # Set the row indices where the velocity condition is True for each column
        vel_filtered_indices[cond_vel_0, 0] = torch.arange(link_vel.shape[0]).to(link_pos.device)[
            cond_vel_0
        ]  # row indices for link_vel[:, 0]
        vel_filtered_indices[cond_vel_1, 1] = torch.arange(link_vel.shape[0]).to(link_pos.device)[
            cond_vel_1
        ]  # row indices for link_vel[:, 1]

        # generate feet contact signal
        # Initialize the feet_contact_signal tensor with shape (N, 2) filled with 0.0
        feet_contact_signal = torch.zeros(link_pos.shape[0], link_pos.shape[1], dtype=torch.float32).to(link_pos.device)

        # Apply the combined condition (both height and velocity conditions must be true)
        cond_combined_0 = cond_0 & cond_vel_0  # Column 0: Both height and velocity conditions
        cond_combined_1 = cond_1 & cond_vel_1  # Column 1: Both height and velocity conditions
        if disable_height:
            cond_combined_0 = cond_vel_0
            cond_combined_1 = cond_vel_1

        # Set the filtered indices where both conditions are true for each column
        feet_contact_signal[cond_combined_0, 0] = 1.0  # For column 0
        feet_contact_signal[cond_combined_1, 1] = 1.0  # For column 1
        feet_contact_indices.append(feet_contact_signal)

        # generate filtered root position (root_pos[:, 2] - root_pos_offset)
        filtered_root_pos = root_pos[i].clone()
        avg_contact_height = (
            torch.sum(link_pos[cond_combined_0, 0, 2]) + torch.sum(link_pos[cond_combined_1, 1, 2])
        ) / (torch.sum(cond_combined_0) + torch.sum(cond_combined_1) + 1e-6)
        sigma_contact_height = torch.std(
            torch.cat([link_pos[cond_combined_0, 0, 2], link_pos[cond_combined_1, 1, 2]], dim=0)
        )
        if torch.isnan(sigma_contact_height).any() or torch.isinf(sigma_contact_height).any():  # replace sigma with 0
            sigma_contact_height = torch.zeros_like(sigma_contact_height)

        contact_height_offset = avg_contact_height + 3 * sigma_contact_height  # 3 sigma rule
        contact_height_offset = torch.clamp(contact_height_offset, min=-0.4, max=0.4)  # avoid too large offset
        filtered_root_pos[:, 2] -= contact_height_offset
        filtered_root_pos[:, 2] = torch.clamp(filtered_root_pos[:, 2], min=0.1)  # Ensure root_pos[:, 2] is non-negative
        filtered_root_poses.append(filtered_root_pos)

        if debug_visualize:
            import os
            import time

            import matplotlib.pyplot as plt
            import numpy as np

            # appendix format: _TIME_HH-MM-SS
            save_rand_appendix = "_TIME_" + time.strftime("%H-%M-%S", time.localtime())
            if debug_save_dir is None:
                save_dir = os.path.join(os.path.dirname(__file__), "debug_math_utils")

            else:
                save_dir = debug_save_dir

            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            link_pos -= min_link_height

            # 4 subplots: link_pos[:,2] with height_filtered_indices, link_vel with vel_filtered_indices, root_pos with root_pos_offset, feet_contact_signal
            fig, axs = plt.subplots(4, 1, figsize=(10, 20))
            axs[0].plot(link_pos[:, 0, 2].cpu().numpy(), label="Link 0 Height")
            axs[0].plot(link_pos[:, 1, 2].cpu().numpy(), label="Link 1 Height")
            axs[0].scatter(
                height_filtered_indices[:, 0].cpu().numpy(),
                link_pos[height_filtered_indices[:, 0], 0, 2].cpu().numpy(),
                color="red",
                label="Filtered Link 0",
            )
            axs[0].scatter(
                height_filtered_indices[:, 1].cpu().numpy(),
                link_pos[height_filtered_indices[:, 1], 1, 2].cpu().numpy(),
                color="green",
                label="Filtered Link 1",
            )
            axs[0].set_title("Link Position")
            axs[0].legend()
            axs[1].plot(link_vel[:, 0].cpu().numpy(), label="Link 0 Velocity")
            axs[1].plot(link_vel[:, 1].cpu().numpy(), label="Link 1 Velocity")
            axs[1].scatter(
                vel_filtered_indices[:, 0].cpu().numpy(),
                link_vel[vel_filtered_indices[:, 0], 0].cpu().numpy(),
                color="red",
                label="Filtered Link 0 Velocity",
            )
            axs[1].scatter(
                vel_filtered_indices[:, 1].cpu().numpy(),
                link_vel[vel_filtered_indices[:, 1], 1].cpu().numpy(),
                color="green",
                label="Filtered Link 1 Velocity",
            )
            axs[1].set_title("Link Velocity")
            axs[1].legend()
            axs[2].plot(root_pos[i][:, 0].cpu().numpy(), label="Root Position X")
            axs[2].plot(root_pos[i][:, 1].cpu().numpy(), label="Root Position Y")
            axs[2].plot(root_pos[i][:, 2].cpu().numpy(), label="Root Position Z")
            axs[2].plot(filtered_root_pos[:, 0].cpu().numpy(), label="Filtered Root Position X", linestyle="--")
            axs[2].plot(filtered_root_pos[:, 1].cpu().numpy(), label="Filtered Root Position Y", linestyle="--")
            axs[2].plot(filtered_root_pos[:, 2].cpu().numpy(), label="Filtered Root Position Z", linestyle="--")
            axs[2].set_title("Root Position")
            axs[2].legend()
            axs[3].plot(feet_contact_signal[:, 0].cpu().numpy(), label="Link 0 Contact Signal")
            axs[3].plot(feet_contact_signal[:, 1].cpu().numpy(), label="Link 1 Contact Signal")
            axs[3].plot(
                np.sin(contact_to_phase(feet_contact_signal[:, 0], 0.55).cpu().numpy() * 2 * np.pi),
                label="Link 0 Phase Signal",
                linestyle="--",
            )
            axs[3].plot(
                np.sin(contact_to_phase(feet_contact_signal[:, 1]).cpu().numpy() * 2 * np.pi),
                label="Link 1 Phase Signal",
                linestyle="--",
            )
            axs[3].set_title("Feet Contact Signal")
            axs[3].legend()
            plt.tight_layout()
            # save name: debug_feet_contact_{i}_{save_rand_appendix}.png
            plt.savefig(os.path.join(save_dir, f"debug_feet_contact_{i}_{save_rand_appendix}.png"))
            plt.close(fig)
            # from PIL import Image
            # visualizer = debug_rv
            # for j in np.int32(np.linspace(0, actions.shape[1], endpoint=False, num=5)):
            # # for j in [0, 2425, 4007, 4008]:
            #     Image.fromarray(visualizer(actions[i][j])).save(os.path.join(save_dir, f'visualize_{i}_{j}.png'))

    feet_contact_indices = torch.stack(feet_contact_indices, dim=0)  # shape (B, N)
    filtered_root_poses = torch.stack(filtered_root_poses, dim=0)  # shape (B, N, 3)
    return feet_contact_indices, filtered_root_poses


@torch.jit.script
def pad_to_len(tensor: torch.Tensor, target_len: int, pad_front: bool = False):
    """
    Pad a tensor to the target length by repeating the first or last element. If the tensor is longer than the target length, it is truncated accordingly.

    Args:
        tensor (torch.Tensor): Input tensor of shape (N, C).
        target_len (int): Target length.
        pad_front (bool): Whether to pad the front or back of the tensor.

    Returns:
        torch.Tensor: Padded tensor of shape (target_len, C).
    """
    # Get the length of the input tensor
    N = tensor.size(0)

    # If the tensor is already at the target length, return as is
    if N == target_len:
        return tensor

    # If the tensor is longer than the target length, truncate it
    if N > target_len:
        return tensor[:target_len] if not pad_front else tensor[-target_len:]

    # Compute the number of elements to pad
    num_pad = target_len - N

    # Repeat the first or last element to pad the tensor
    if pad_front:
        return torch.cat([tensor[0].unsqueeze(0).repeat(num_pad, 1), tensor], dim=0)
    else:
        return torch.cat([tensor, tensor[-1].unsqueeze(0).repeat(num_pad, 1)], dim=0)


@torch.jit.script
def batch_pad_to_len(tensor: torch.Tensor, target_len: int, pad_front: bool = False):
    """
    Pad a batched tensor to the target length by repeating the first or last element. If the tensor is longer than the target length, it is truncated accordingly.

    Args:
        tensor (torch.Tensor): Input tensor of shape (B, N, C).
        target_len (int): Target length.
        pad_front (bool): Whether to pad the front or back of the tensor.

    Returns:
        torch.Tensor: Padded tensor of shape (B, target_len, C).
    """
    # Get the length of the input tensor
    N = tensor.size(1)

    # If the tensor is already at the target length, return as is
    if N == target_len:
        return tensor

    # If the tensor is longer than the target length, truncate it
    if N > target_len:
        return tensor[:, :target_len] if not pad_front else tensor[:, -target_len:]

    # Compute the number of elements to pad
    num_pad = target_len - N

    # Repeat the first or last element to pad the tensor
    if pad_front:
        return torch.cat([tensor[:, 0].unsqueeze(1).repeat(1, num_pad, 1), tensor], dim=1)
    else:
        return torch.cat([tensor, tensor[:, -1].unsqueeze(1).repeat(1, num_pad, 1)], dim=1)


"""
Rotation (From Isaac Lab)
"""


@torch.jit.script
def normalize(x: torch.Tensor, eps: float = 1e-9) -> torch.Tensor:
    """Normalizes a given input tensor to unit length.

    Args:
        x: Input tensor of shape (N, dims).
        eps: A small value to avoid division by zero. Defaults to 1e-9.

    Returns:
        Normalized tensor of shape (N, dims).
    """
    return x / x.norm(p=2, dim=-1).clamp(min=eps, max=None).unsqueeze(-1)


@torch.jit.script
def wrap_to_pi(angles: torch.Tensor) -> torch.Tensor:
    r"""Wraps input angles (in radians) to the range :math:`[-\pi, \pi]`.

    This function wraps angles in radians to the range :math:`[-\pi, \pi]`, such that
    :math:`\pi` maps to :math:`\pi`, and :math:`-\pi` maps to :math:`-\pi`. In general,
    odd positive multiples of :math:`\pi` are mapped to :math:`\pi`, and odd negative
    multiples of :math:`\pi` are mapped to :math:`-\pi`.

    The function behaves similar to MATLAB's `wrapToPi <https://www.mathworks.com/help/map/ref/wraptopi.html>`_
    function.

    Args:
        angles: Input angles of any shape.

    Returns:
        Angles in the range :math:`[-\pi, \pi]`.
    """
    # wrap to [0, 2*pi)
    wrapped_angle = (angles + torch.pi) % (2 * torch.pi)
    # map to [-pi, pi]
    # we check for zero in wrapped angle to make it go to pi when input angle is odd multiple of pi
    return torch.where((wrapped_angle == 0) & (angles > 0), torch.pi, wrapped_angle - torch.pi)


@torch.jit.script
def matrix_from_quat(quaternions: torch.Tensor) -> torch.Tensor:
    """Convert rotations given as quaternions to rotation matrices.

    Args:
        quaternions: The quaternion orientation in (w, x, y, z). Shape is (..., 4).

    Returns:
        Rotation matrices. The shape is (..., 3, 3).

    Reference:
        https://github.com/facebookresearch/pytorch3d/blob/main/pytorch3d/transforms/rotation_conversions.py#L41-L70
    """
    r, i, j, k = torch.unbind(quaternions, -1)
    # pyre-fixme[58]: `/` is not supported for operand types `float` and `Tensor`.
    two_s = 2.0 / (quaternions * quaternions).sum(-1)

    o = torch.stack(
        (
            1 - two_s * (j * j + k * k),
            two_s * (i * j - k * r),
            two_s * (i * k + j * r),
            two_s * (i * j + k * r),
            1 - two_s * (i * i + k * k),
            two_s * (j * k - i * r),
            two_s * (i * k - j * r),
            two_s * (j * k + i * r),
            1 - two_s * (i * i + j * j),
        ),
        -1,
    )
    return o.reshape(quaternions.shape[:-1] + (3, 3))


def convert_quat(quat: torch.Tensor | np.ndarray, to: Literal["xyzw", "wxyz"] = "xyzw") -> torch.Tensor | np.ndarray:
    """Converts quaternion from one convention to another.

    The convention to convert TO is specified as an optional argument. If to == 'xyzw',
    then the input is in 'wxyz' format, and vice-versa.

    Args:
        quat: The quaternion of shape (..., 4).
        to: Convention to convert quaternion to.. Defaults to "xyzw".

    Returns:
        The converted quaternion in specified convention.

    Raises:
        ValueError: Invalid input argument `to`, i.e. not "xyzw" or "wxyz".
        ValueError: Invalid shape of input `quat`, i.e. not (..., 4,).
    """
    # check input is correct
    if quat.shape[-1] != 4:
        msg = f"Expected input quaternion shape mismatch: {quat.shape} != (..., 4)."
        raise ValueError(msg)
    if to not in ["xyzw", "wxyz"]:
        msg = f"Expected input argument `to` to be 'xyzw' or 'wxyz'. Received: {to}."
        raise ValueError(msg)
    # check if input is numpy array (we support this backend since some classes use numpy)
    if isinstance(quat, np.ndarray):
        # use numpy functions
        if to == "xyzw":
            # wxyz -> xyzw
            return np.roll(quat, -1, axis=-1)
        else:
            # xyzw -> wxyz
            return np.roll(quat, 1, axis=-1)
    else:
        # convert to torch (sanity check)
        if not isinstance(quat, torch.Tensor):
            quat = torch.tensor(quat, dtype=float)
        # convert to specified quaternion type
        if to == "xyzw":
            # wxyz -> xyzw
            return quat.roll(-1, dims=-1)
        else:
            # xyzw -> wxyz
            return quat.roll(1, dims=-1)


@torch.jit.script
def quat_conjugate(q: torch.Tensor) -> torch.Tensor:
    """Computes the conjugate of a quaternion.

    Args:
        q: The quaternion orientation in (w, x, y, z). Shape is (..., 4).

    Returns:
        The conjugate quaternion in (w, x, y, z). Shape is (..., 4).
    """
    shape = q.shape
    q = q.reshape(-1, 4)
    return torch.cat((q[:, 0:1], -q[:, 1:]), dim=-1).view(shape)


@torch.jit.script
def quat_inv(q: torch.Tensor) -> torch.Tensor:
    """Compute the inverse of a quaternion.

    Args:
        q: The quaternion orientation in (w, x, y, z). Shape is (N, 4).

    Returns:
        The inverse quaternion in (w, x, y, z). Shape is (N, 4).
    """
    return normalize(quat_conjugate(q))


@torch.jit.script
def quat_from_euler_xyz(roll: torch.Tensor, pitch: torch.Tensor, yaw: torch.Tensor) -> torch.Tensor:
    """Convert rotations given as Euler angles in radians to Quaternions.

    Note:
        The euler angles are assumed in XYZ convention.

    Args:
        roll: Rotation around x-axis (in radians). Shape is (N,).
        pitch: Rotation around y-axis (in radians). Shape is (N,).
        yaw: Rotation around z-axis (in radians). Shape is (N,).

    Returns:
        The quaternion in (w, x, y, z). Shape is (N, 4).
    """
    cy = torch.cos(yaw * 0.5)
    sy = torch.sin(yaw * 0.5)
    cr = torch.cos(roll * 0.5)
    sr = torch.sin(roll * 0.5)
    cp = torch.cos(pitch * 0.5)
    sp = torch.sin(pitch * 0.5)
    # compute quaternion
    qw = cy * cr * cp + sy * sr * sp
    qx = cy * sr * cp - sy * cr * sp
    qy = cy * cr * sp + sy * sr * cp
    qz = sy * cr * cp - cy * sr * sp

    return torch.stack([qw, qx, qy, qz], dim=-1)


@torch.jit.script
def _sqrt_positive_part(x: torch.Tensor) -> torch.Tensor:
    """Returns torch.sqrt(torch.max(0, x)) but with a zero sub-gradient where x is 0.

    Reference:
        https://github.com/facebookresearch/pytorch3d/blob/main/pytorch3d/transforms/rotation_conversions.py#L91-L99
    """
    ret = torch.zeros_like(x)
    positive_mask = x > 0
    ret[positive_mask] = torch.sqrt(x[positive_mask])
    return ret


@torch.jit.script
def quat_from_matrix(matrix: torch.Tensor) -> torch.Tensor:
    """Convert rotations given as rotation matrices to quaternions.

    Args:
        matrix: The rotation matrices. Shape is (..., 3, 3).

    Returns:
        The quaternion in (w, x, y, z). Shape is (..., 4).

    Reference:
        https://github.com/facebookresearch/pytorch3d/blob/main/pytorch3d/transforms/rotation_conversions.py#L102-L161
    """
    if matrix.size(-1) != 3 or matrix.size(-2) != 3:
        raise ValueError(f"Invalid rotation matrix shape {matrix.shape}.")

    batch_dim = matrix.shape[:-2]
    m00, m01, m02, m10, m11, m12, m20, m21, m22 = torch.unbind(matrix.reshape(batch_dim + (9,)), dim=-1)

    q_abs = _sqrt_positive_part(
        torch.stack(
            [
                1.0 + m00 + m11 + m22,
                1.0 + m00 - m11 - m22,
                1.0 - m00 + m11 - m22,
                1.0 - m00 - m11 + m22,
            ],
            dim=-1,
        )
    )

    # we produce the desired quaternion multiplied by each of r, i, j, k
    quat_by_rijk = torch.stack(
        [
            # pyre-fixme[58]: `**` is not supported for operand types `Tensor` and `int`.
            torch.stack([q_abs[..., 0] ** 2, m21 - m12, m02 - m20, m10 - m01], dim=-1),
            # pyre-fixme[58]: `**` is not supported for operand types `Tensor` and `int`.
            torch.stack([m21 - m12, q_abs[..., 1] ** 2, m10 + m01, m02 + m20], dim=-1),
            # pyre-fixme[58]: `**` is not supported for operand types `Tensor` and `int`.
            torch.stack([m02 - m20, m10 + m01, q_abs[..., 2] ** 2, m12 + m21], dim=-1),
            # pyre-fixme[58]: `**` is not supported for operand types `Tensor` and `int`.
            torch.stack([m10 - m01, m20 + m02, m21 + m12, q_abs[..., 3] ** 2], dim=-1),
        ],
        dim=-2,
    )

    # We floor here at 0.1 but the exact level is not important; if q_abs is small,
    # the candidate won't be picked.
    flr = torch.tensor(0.1).to(dtype=q_abs.dtype, device=q_abs.device)
    quat_candidates = quat_by_rijk / (2.0 * q_abs[..., None].max(flr))

    # if not for numerical problems, quat_candidates[i] should be same (up to a sign),
    # forall i; we pick the best-conditioned one (with the largest denominator)
    return quat_candidates[torch.nn.functional.one_hot(q_abs.argmax(dim=-1), num_classes=4) > 0.5, :].reshape(
        batch_dim + (4,)
    )


@torch.jit.script
def quat_apply(quat: torch.Tensor, vec: torch.Tensor) -> torch.Tensor:
    """Apply a quaternion rotation to a vector.

    Args:
        quat: The quaternion in (w, x, y, z). Shape is (..., 4).
        vec: The vector in (x, y, z). Shape is (..., 3).

    Returns:
        The rotated vector in (x, y, z). Shape is (..., 3).
    """
    # store shape
    shape = vec.shape
    # reshape to (N, 3) for multiplication
    quat = quat.reshape(-1, 4)
    vec = vec.reshape(-1, 3)
    # extract components from quaternions
    xyz = quat[:, 1:]
    t = xyz.cross(vec, dim=-1) * 2
    return (vec + quat[:, 0:1] * t + xyz.cross(t, dim=-1)).view(shape)


@torch.jit.script
def yaw_quat(quat: torch.Tensor) -> torch.Tensor:
    """Extract the yaw component of a quaternion.

    Args:
        quat: The orientation in (w, x, y, z). Shape is (..., 4)

    Returns:
        A quaternion with only yaw component.
    """
    shape = quat.shape
    quat_yaw = quat.clone().view(-1, 4)
    qw = quat_yaw[:, 0]
    qx = quat_yaw[:, 1]
    qy = quat_yaw[:, 2]
    qz = quat_yaw[:, 3]
    yaw = torch.atan2(2 * (qw * qz + qx * qy), 1 - 2 * (qy * qy + qz * qz))
    quat_yaw[:] = 0.0
    quat_yaw[:, 3] = torch.sin(yaw / 2)
    quat_yaw[:, 0] = torch.cos(yaw / 2)
    quat_yaw = normalize(quat_yaw)
    return quat_yaw.view(shape)


@torch.jit.script
def quat_apply_yaw(quat: torch.Tensor, vec: torch.Tensor) -> torch.Tensor:
    """Rotate a vector only around the yaw-direction.

    Args:
        quat: The orientation in (w, x, y, z). Shape is (N, 4).
        vec: The vector in (x, y, z). Shape is (N, 3).

    Returns:
        The rotated vector in (x, y, z). Shape is (N, 3).
    """
    quat_yaw = yaw_quat(quat)
    return quat_apply(quat_yaw, vec)


@torch.jit.script
def quat_rotate(q: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """Rotate a vector by a quaternion along the last dimension of q and v.

    Args:
        q: The quaternion in (w, x, y, z). Shape is (..., 4).
        v: The vector in (x, y, z). Shape is (..., 3).

    Returns:
        The rotated vector in (x, y, z). Shape is (..., 3).
    """
    q_w = q[..., 0]
    q_vec = q[..., 1:]
    a = v * (2.0 * q_w**2 - 1.0).unsqueeze(-1)
    b = torch.cross(q_vec, v, dim=-1) * q_w.unsqueeze(-1) * 2.0
    # for two-dimensional tensors, bmm is faster than einsum
    if q_vec.dim() == 2:
        c = q_vec * torch.bmm(q_vec.view(q.shape[0], 1, 3), v.view(q.shape[0], 3, 1)).squeeze(-1) * 2.0
    else:
        c = q_vec * torch.einsum("...i,...i->...", q_vec, v).unsqueeze(-1) * 2.0
    return a + b + c


@torch.jit.script
def quat_rotate_inverse(q: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """Rotate a vector by the inverse of a quaternion along the last dimension of q and v.

    Args:
        q: The quaternion in (w, x, y, z). Shape is (..., 4).
        v: The vector in (x, y, z). Shape is (..., 3).

    Returns:
        The rotated vector in (x, y, z). Shape is (..., 3).
    """
    q_shape = q.shape
    v_shape = v.shape
    if 0 in q_shape or 0 in v_shape:
        return v
    q_w = q[..., 0]
    q_vec = q[..., 1:]
    a = v * (2.0 * q_w**2 - 1.0).unsqueeze(-1)
    b = torch.cross(q_vec, v, dim=-1) * q_w.unsqueeze(-1) * 2.0
    # for two-dimensional tensors, bmm is faster than einsum
    if q_vec.dim() == 2:
        c = q_vec * torch.bmm(q_vec.view(q.shape[0], 1, 3), v.view(q.shape[0], 3, 1)).squeeze(-1) * 2.0
    else:
        c = q_vec * torch.einsum("...i,...i->...", q_vec, v).unsqueeze(-1) * 2.0
    return a - b + c

def quat_rotate_inverse_non_jit(q: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """Rotate a vector by the inverse of a quaternion along the last dimension of q and v.

    Args:
        q: The quaternion in (w, x, y, z). Shape is (..., 4).
        v: The vector in (x, y, z). Shape is (..., 3).

    Returns:
        The rotated vector in (x, y, z). Shape is (..., 3).
    """
    q_w = q[..., 0]
    q_vec = q[..., 1:]
    a = v * (2.0 * q_w**2 - 1.0).unsqueeze(-1)
    b = torch.cross(q_vec, v, dim=-1) * q_w.unsqueeze(-1) * 2.0
    # for two-dimensional tensors, bmm is faster than einsum
    if q_vec.dim() == 2:
        c = q_vec * torch.bmm(q_vec.view(q.shape[0], 1, 3), v.view(q.shape[0], 3, 1)).squeeze(-1) * 2.0
    else:
        c = q_vec * torch.einsum("...i,...i->...", q_vec, v).unsqueeze(-1) * 2.0
    return a - b + c


@torch.jit.script
def unwrap_angle(angles: torch.Tensor, period: float = 2 * np.pi) -> torch.Tensor:
    """
    Flatten a 1D tensor of angles (in radians) to create a continuous sequence of angles.

    Args:
        angles (torch.Tensor): 1D tensor of angles in radians, shape (N,).
        period (float): Period of the angles, default is 2*pi.

    Returns:
        torch.Tensor: Unwrapped continuous angle sequence.
    """
    unwrapped = angles.clone()
    # Compute the difference between adjacent angles
    diff = torch.diff(angles)
    #   # Detect jumps in the angle sequence
    # From ~2pi to ~0 (diff < -pi)
    jumps_up = diff < -period / 2
    # From ~0 to ~2pi (diff > pi)
    jumps_down = diff > period / 2

    # Accumulate corrections
    corrections = torch.zeros_like(angles)
    corrections[1:] += torch.cumsum(jumps_up.to(angles.dtype), dim=0) * period
    corrections[1:] -= torch.cumsum(jumps_down.to(angles.dtype), dim=0) * period
    return unwrapped + corrections


@torch.jit.script
def simple_moving_average(x: torch.Tensor, window_size: int) -> torch.Tensor:
    """
    Apply moving average to 2D tensors
    Args:
        x (torch.Tensor): input tensor (T, D), where T is the sequence length and D is the number of features.
        window_size (int): Size of the moving average window.

    Returns:
        torch.Tensor: Smoothed tensor (T, D)
    """
    if window_size <= 1:
        return x
    
    T, D = x.shape
    x_reshaped = x.t().unsqueeze(0)

    weight = torch.ones(D, 1, window_size, device=x.device, dtype=x.dtype) / window_size

    padding = window_size // 2
    smoothed_reshaped = torch.nn.functional.conv1d(x_reshaped, weight, padding=padding, groups=D)
    return smoothed_reshaped.squeeze(0).t()

@torch.jit.script
def unwrap_and_smooth_rot_vecs(rot_vecs: torch.Tensor, smoothing_window: int = 5) -> torch.Tensor:
    """
    Unwrap and smooth a sequence of rotation vectors.

    Args:
        rot_vecs (torch.Tensor): Input rotation vector sequence (T, 3).
        smoothing_window (int): Size of the moving average window.

    Returns:
        torch.Tensor: Fixed rotation vector sequence (T, 3).
    """
    # 1. Decompose into angles and axes
    angles = torch.norm(rot_vecs, dim=1)
    # Add a small epsilon to prevent division by zero
    axes = rot_vecs / (angles.unsqueeze(1) + 1e-9)

    # 2. Unwrap angle sequence
    unwrapped_angles = unwrap_angle(angles)

    # 3. Smooth the unwrapped angles
    smoothed_unwrapped_angles = simple_moving_average(unwrapped_angles.unsqueeze(1), smoothing_window).squeeze(1)
    # Smooth the axes (note: need to renormalize after smoothing)
    smoothed_axes = simple_moving_average(axes, smoothing_window)
    smoothed_axes = torch.nn.functional.normalize(smoothed_axes, p=2.0, dim=1)

    # 4. Reconstruct the rotation vector
    fixed_rot_vecs = smoothed_unwrapped_angles.unsqueeze(1) * smoothed_axes

    return fixed_rot_vecs

def simple_moving_average_quat(q: torch.Tensor, window_size: int) -> torch.Tensor:
    """
    Applies a simple moving average filter to a sequence of quaternions.
    
    This function smooths each of the 4 quaternion components independently.
    It uses a 1D convolution to efficiently compute the moving average.

    Args:
        q (torch.Tensor): A tensor of quaternions with shape [T, 4], 
                          where T is the number of frames (time).
        window_size (int): The size of the moving average window. Must be an odd number
                           for a centered average.

    Returns:
        torch.Tensor: The smoothed tensor of quaternions, same shape as input [T, 4].
    """
    # --- Input Validation ---
    if window_size <= 1:
        return q
    if window_size > q.shape[0]:
        raise ValueError("window_size cannot be larger than the sequence length.")
    if window_size % 2 == 0:
        print(f"Warning: window_size {window_size} is even. For a centered moving average, an odd window size is recommended.")

    # --- Reshape for Convolution ---
    # conv1d expects input of shape (N, C, L), where N is batch size, C is channels, L is length.
    # Our input q is [T, 4], which corresponds to (Length, Channels).
    # We transpose it to [4, T] and add a batch dimension to get [1, 4, T].
    q_conv_format = q.transpose(0, 1).unsqueeze(0)  # Shape: [1, 4, T]

    # --- Create the Averaging Kernel ---
    # The kernel will average the values in the window.
    # To process each of the 4 quaternion channels independently, we use 'groups=4'.
    # This means we need a kernel of shape (out_channels, in_channels/groups, kernel_width)
    # which is (4, 1, window_size) for our case.
    kernel = torch.ones(4, 1, window_size, device=q.device, dtype=q.dtype) / window_size

    # --- Apply Convolution ---
    # 'padding="same"' ensures the output length is the same as the input length (T).
    smoothed_q_conv = F.conv1d(q_conv_format, weight=kernel, padding='same', groups=4)

    # --- Reshape Back ---
    # Squeeze the batch dimension and transpose back to the original [T, 4] format.
    smoothed_q = smoothed_q_conv.squeeze(0).transpose(0, 1) # Shape: [T, 4]
    
    return smoothed_q


def smooth_quat_savgol(q: torch.Tensor, window_size: int, polyorder: int) -> torch.Tensor:
    """
    Applies a Savitzky-Golay filter to a sequence of quaternions.

    This function is a wrapper around scipy's savgol_filter. It's generally
    superior to a simple moving average as it fits a polynomial to the data,
    which can better preserve the signal's features while removing noise.

    NOTE: This function involves a CPU roundtrip (Tensor -> NumPy -> Tensor) and
    will break the computation graph for autograd. It is intended for data
    pre-processing or post-processing, not for use within a differentiable model.

    Args:
        q (torch.Tensor): A tensor of quaternions with shape [T, 4].
        window_size (int): The length of the filter window. Must be a positive odd integer.
        polyorder (int): The order of the polynomial used to fit the samples.
                         Must be less than window_size.

    Returns:
        torch.Tensor: The smoothed tensor of quaternions, same shape as input [T, 4].
    """
    # --- Input Validation ---
    if polyorder >= window_size:
        raise ValueError("polyorder must be less than window_size.")
    if window_size % 2 == 0:
        raise ValueError("window_size must be an odd number.")
    if window_size > q.shape[0]:
        print(f"Warning: window_size ({window_size}) is > sequence length ({q.shape[0]}). "
              f"Filter will not be applied.")
        return q
        
    # --- Convert to NumPy on CPU ---
    # Scipy operates on NumPy arrays, and the data must be on the CPU.
    original_device = q.device
    q_numpy = q.cpu().numpy()

    # --- Apply Savitzky-Golay filter ---
    # We apply the filter along the time axis (axis=0).
    smoothed_q_numpy = savgol_filter(
        q_numpy, 
        window_length=window_size, 
        polyorder=polyorder, 
        axis=0
    )

    # --- Convert back to PyTorch Tensor on the original device ---
    # Ensure the new tensor has the same dtype and is on the same device as the input.
    smoothed_q = torch.from_numpy(smoothed_q_numpy).to(original_device).type(q.dtype)
    
    return smoothed_q

def is_foot_parallel_from_rot_matrix(rot_mat_batch: torch.Tensor, tolerance_deg: float = 5.0) -> torch.Tensor:
    """
    (Optimized Version)
    Checks if a batch of robot feet are parallel to the ground using rotation matrices.
    This version avoids expensive acos calculation by comparing cosines directly.

    Args:
        rot_mat_batch (torch.Tensor): A tensor of rotation matrices with shape (B, 2, 3, 3).
                                      B is the batch size, 2 represents the left and right foot.
        tolerance_deg (float): The maximum acceptable deviation angle in degrees.

    Returns:
        torch.Tensor: A boolean tensor of shape (B, 2).
                      True if the corresponding foot is parallel, False otherwise.
    """
    device = rot_mat_batch.device
    dtype = rot_mat_batch.dtype
    # --- 1. Define Base Vector ---
    foot_local_normal = torch.tensor([0.0, 0.0, 1.0], device=device, dtype=dtype)
    # --- 2. Rotate the Foot's Normal Vector ---
    foot_world_normal = torch.matmul(rot_mat_batch, foot_local_normal.view(1, 1, 3, 1)).squeeze(-1)
    # --- 3. Get the Dot Product ---
    # The dot product with the world's up-vector [0,0,1] is simply the Z-component.
    dot_product = foot_world_normal[..., 2]
    # --- 4. Compare Cosines Directly (Optimization) ---
    # Pre-calculate the cosine of the tolerance angle.
    # We only need to do this once.
    tolerance_rad = torch.deg2rad(torch.tensor(tolerance_deg, device=device, dtype=dtype))
    cos_tolerance = torch.cos(tolerance_rad)
    # Check if the absolute value of the dot product is greater than or equal to the cosine threshold.
    # This is mathematically equivalent to checking if the angle is close to 0 OR 180 degrees,
    # but is computationally much cheaper than using acos.
    is_parallel = torch.abs(dot_product) >= cos_tolerance
    return is_parallel
