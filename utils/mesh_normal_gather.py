"""Face-normal image gathers with the original differentiable normal inputs.

The compact route omits masked pixels before advanced indexing, avoiding their
large duplicate-index run in CUDA indexing backward. Caller sanitation of
pix_to_face (including overflow handling) remains unchanged.
"""
import torch


def _validate_shapes(face_normals, pix_to_face, valid_mask):
    if face_normals.ndim != 2 or face_normals.shape[1] != 3:
        raise ValueError("face_normals must have shape (F, 3)")
    if not face_normals.is_floating_point():
        raise ValueError("face_normals must use a floating dtype")
    if pix_to_face.dtype not in (torch.int32, torch.int64):
        raise ValueError("pix_to_face must use int32 or int64")
    if valid_mask.dtype != torch.bool or valid_mask.shape != pix_to_face.shape:
        raise ValueError("valid_mask must be boolean with the same pixel shape")
    if face_normals.device != pix_to_face.device or face_normals.device != valid_mask.device:
        raise ValueError("normal/index/mask tensors must be on the same device")


def gather_face_normals_legacy(face_normals, pix_to_face, valid_mask):
    """Reference full-image gather followed by the existing mask multiply."""
    _validate_shapes(face_normals, pix_to_face, valid_mask)
    return face_normals[pix_to_face] * valid_mask[..., None]


def gather_face_normals_compact(face_normals, pix_to_face, valid_mask):
    """Gather only valid pixels, then restore zero background in original shape.

    Inputs have the same finite-normal/caller-sanitized-index domain as the
    production renderer. No normal is detached or normalized here. The empty
    advanced gather and index_copy retain a backward path with zero normal
    gradients when all pixels are background, including an empty face array.
    """
    _validate_shapes(face_normals, pix_to_face, valid_mask)
    pixel_indices = valid_mask.reshape(-1).nonzero(as_tuple=False).reshape(-1)
    face_indices = pix_to_face.reshape(-1).index_select(0, pixel_indices)
    # Keep advanced indexing, so the valid-face reduction uses the same Torch
    # indexing backward family/order as the reference rather than index_add.
    gathered = face_normals[face_indices]
    image = face_normals.new_zeros((pix_to_face.numel(), 3))
    image = image.index_copy(0, pixel_indices, gathered)
    return image.reshape(*pix_to_face.shape, 3)
