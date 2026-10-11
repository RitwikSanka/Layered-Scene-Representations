import numpy as np
import cv2
import torch
from PIL import Image
from transformers import pipeline

class BaselineCPipeline:
    """
    Baseline C: Depth-Stratified Multi-Plane Representation
    Continuous depth-stratified scene decomposition:
    - Infers continuous metric/relative depth using Depth-Anything-V2.
    - Quantizes depth space into stratified depth planes (Background, Midground, Foreground).
    - Refines spatial layers via morphological connected-component clustering.
    - Disentangles intrinsic appearance (Albedo and Shading) per depth layer.
    """
    def __init__(self, device="cpu", use_mocks=False):
        self.device = device
        self.use_mocks = use_mocks
        
        print("Initializing Baseline C Depth-Stratified Pipeline...")
        self.depth_estimator = pipeline(
            "depth-estimation", 
            model="depth-anything/Depth-Anything-V2-Small-hf", 
            device=0 if device == "cuda" and torch.cuda.is_available() else -1
        )
        print("Baseline C Pipeline Ready.")

    def _intrinsic_decomposition(self, image_arr):
        """Bilateral Retinex decomposition: I = Albedo x Shading"""
        img_f = image_arr.astype(np.float32) / 255.0
        shading = cv2.bilateralFilter(image_arr, d=9, sigmaColor=75, sigmaSpace=75).astype(np.float32) / 255.0
        shading = np.clip(shading, 0.05, 1.0)
        albedo = np.clip(img_f / (shading + 1e-4), 0.0, 1.0)
        return albedo, shading

    def process(self, image_arr):
        """
        Args:
            image_arr: np.ndarray (H, W, 3) uint8
        Returns:
            List of dicts: 'rgba', 'albedo', 'shading', 'depth', 'label' ordered back to front.
        """
        H, W, _ = image_arr.shape
        image_pil = Image.fromarray(image_arr)
        
        # 1. Monocular Depth
        res = self.depth_estimator(image_pil)
        depth_map = np.array(res["depth"], dtype=np.float32)
        d_min, d_max = depth_map.min(), depth_map.max()
        if d_max > d_min:
            norm_depth = (depth_map - d_min) / (d_max - d_min)
        else:
            norm_depth = np.zeros_like(depth_map)
            
        # 2. Intrinsic decomposition
        albedo_full, shading_full = self._intrinsic_decomposition(image_arr)
        img_f = image_arr.astype(np.float32) / 255.0
        
        # 3. Depth-Stratified Clustering: Far (Background), Mid (Midground), Near (Foreground)
        tiers = [
            (0.0, 0.33, "Background"),
            (0.33, 0.66, "Midground"),
            (0.66, 1.01, "Foreground")
        ]
        
        layers = []
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        
        for q_low, q_high, tier_name in tiers:
            tier_mask = (norm_depth >= q_low) & (norm_depth < q_high)
            
            # Morphological cleaning to ensure contiguous solid regions
            tier_mask_u8 = cv2.morphologyEx(tier_mask.astype(np.uint8), cv2.MORPH_CLOSE, kernel)
            
            if np.sum(tier_mask_u8) < int(H * W * 0.01):
                continue
                
            mask_3d = np.expand_dims(tier_mask_u8.astype(np.float32), axis=-1)
            rgba = np.concatenate([img_f, mask_3d], axis=-1)
            
            layer_albedo = albedo_full * mask_3d
            layer_shading = shading_full * mask_3d
            mean_d = float(norm_depth[tier_mask_u8 > 0].mean()) if np.any(tier_mask_u8 > 0) else (q_low + q_high)/2.0
            
            layers.append({
                'rgba': rgba,
                'albedo': layer_albedo,
                'shading': layer_shading,
                'depth': mean_d,
                'label': f"{tier_name}_Plane"
            })
            
        # Ensure back-to-front ordering (lowest depth = farthest = back)
        layers.sort(key=lambda x: x['depth'])
        
        # Fallback if an image is completely flat
        if not layers:
            full_mask = np.ones((H, W, 1), dtype=np.float32)
            layers.append({
                'rgba': np.concatenate([img_f, full_mask], axis=-1),
                'albedo': albedo_full,
                'shading': shading_full,
                'depth': 0.5,
                'label': "Unified_Scene_Plane"
            })
            
        return layers

if __name__ == "__main__":
    pipeline = BaselineCPipeline()
    test_img = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
    res = pipeline.process(test_img)
    print(f"Baseline C completed with {len(res)} layers.")
