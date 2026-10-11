import numpy as np
import cv2

def composite_layers(layers):
    """
    Composites a list of RGBA layers from back to front.
    Args:
        layers: List of np.ndarray of shape (H, W, 4) in float [0, 1] format, ordered from back to front.
    Returns:
        np.ndarray of shape (H, W, 3) in float [0, 1] format.
    """
    if not layers:
        return np.zeros((1, 1, 3))
        
    H, W = layers[0].shape[:2]
    result = np.zeros((H, W, 3), dtype=np.float32)
    alpha_acc = np.zeros((H, W, 1), dtype=np.float32)
    
    # Back to front compositing
    for layer in layers:
        rgb = layer[..., :3]
        alpha = layer[..., 3:4]
        
        # Standard alpha blending
        result = rgb * alpha + result * (1 - alpha)
        alpha_acc = alpha + alpha_acc * (1 - alpha)
        
    return result

def apply_intrinsics(albedo, shading):
    """
    Combines albedo and shading to reconstruct the RGB layer.
    Retinex assumption: I = albedo * shading
    """
    # Assuming shading is 1-channel or 3-channel
    if shading.shape[-1] == 1:
        shading = np.repeat(shading, 3, axis=-1)
    
    return albedo * shading
