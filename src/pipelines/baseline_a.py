import numpy as np
import cv2
from skimage.segmentation import slic
from skimage.color import rgb2lab, lab2rgb
from scipy.ndimage import gaussian_filter

class BaselineAPipeline:
    def __init__(self, num_segments=10, compactness=10):
        self.num_segments = num_segments
        self.compactness = compactness

    def _retinex_decomposition(self, image_rgb):
        """
        Simple retinex-like decomposition using Gaussian blur.
        Shading is low-frequency, Albedo is high-frequency.
        """
        # Convert to float
        img_f = image_rgb.astype(np.float32) / 255.0
        
        # Shading approximation (illumination)
        shading = gaussian_filter(img_f, sigma=15)
        
        # Albedo approximation (reflectance)
        # Avoid division by zero
        albedo = img_f / (shading + 1e-5)
        albedo = np.clip(albedo, 0, 1)
        
        return albedo, shading

    def process(self, image):
        """
        Processes an RGB image into layers.
        Args:
            image: np.ndarray (H, W, 3) uint8
        Returns:
            List of dicts containing 'rgba', 'albedo', 'shading', 'depth', 'label'
        """
        H, W, _ = image.shape
        
        # 1. Superpixel segmentation
        segments = slic(image, n_segments=self.num_segments, compactness=self.compactness, start_label=1)
        
        # 2. Retinex
        albedo_img, shading_img = self._retinex_decomposition(image)
        
        layers = []
        unique_labels = np.unique(segments)
        
        for label in unique_labels:
            mask = (segments == label).astype(np.float32)
            
            # Find centroid for vertical depth prior
            y_indices, x_indices = np.where(segments == label)
            if len(y_indices) == 0:
                continue
                
            centroid_y = np.mean(y_indices)
            
            # Depth ordering: objects lower in the image (higher y) are generally closer
            # Let's say depth is H - centroid_y (so higher y = smaller depth = closer)
            depth_score = centroid_y 
            
            # Extract layer
            mask_3d = np.expand_dims(mask, axis=-1)
            rgba = np.concatenate([image.astype(np.float32) / 255.0, mask_3d], axis=-1)
            
            # Mask intrinsic components
            layer_albedo = albedo_img * mask_3d
            layer_shading = shading_img * mask_3d
            
            layers.append({
                'rgba': rgba,
                'albedo': layer_albedo,
                'shading': layer_shading,
                'depth': depth_score,
                'label': f"Superpixel_{label}"
            })
            
        # Sort layers from back (low y) to front (high y)
        layers.sort(key=lambda x: x['depth'])
        
        return layers

if __name__ == "__main__":
    pipeline = BaselineAPipeline()
    dummy_img = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
    res = pipeline.process(dummy_img)
    print(f"Baseline A generated {len(res)} layers.")
