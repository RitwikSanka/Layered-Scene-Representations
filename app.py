import gradio as gr
import numpy as np
import cv2
from PIL import Image

from src.pipelines.baseline_a import BaselineAPipeline
from src.pipelines.baseline_b import BaselineBPipeline
from src.pipelines.baseline_c import BaselineCPipeline

# Initialize pipelines globally to avoid reloading
pipelines = {
    'Baseline A': BaselineAPipeline(),
    'Baseline B': BaselineBPipeline(use_mocks=False),
    'Baseline C': BaselineCPipeline(use_mocks=False)
}

def process_image(image, model_choice):
    if image is None:
        return None, None, None
    
    # Convert PIL to numpy
    img_arr = np.array(image)
    
    pipeline = pipelines[model_choice]
    layers = pipeline.process(img_arr)
    
    # Extract components for viewing
    layer_imgs = []
    albedo_imgs = []
    shading_imgs = []
    
    for idx, layer in enumerate(layers):
        # Convert to uint8 for Gradio
        rgba = (layer['rgba'] * 255).astype(np.uint8)
        # Pre-multiply alpha for better visualization against white/black background
        rgb = rgba[..., :3]
        alpha = rgba[..., 3:4] / 255.0
        vis_rgba = (rgb * alpha + 255 * (1 - alpha)).astype(np.uint8)
        
        layer_imgs.append((vis_rgba, layer['label']))
        
        albedo = (layer['albedo'] * 255).astype(np.uint8)
        shading = (layer['shading'] * 255).astype(np.uint8)
        
        albedo_imgs.append((albedo, layer['label']))
        shading_imgs.append((shading, layer['label']))
        
    return layer_imgs, albedo_imgs, shading_imgs

def generate_parallax(image, model_choice, offset_x, offset_y):
    """
    Simulates a 2.5D parallax effect by shifting layers based on depth.
    """
    if image is None:
        return None
        
    img_arr = np.array(image)
    pipeline = pipelines[model_choice]
    layers = pipeline.process(img_arr)
    
    H, W = img_arr.shape[:2]
    result = np.zeros((H, W, 3), dtype=np.float32)
    alpha_acc = np.zeros((H, W, 1), dtype=np.float32)
    
    # Assume depths are scaled roughly 0 to max_depth
    # We want front layers to move more than back layers
    if not layers:
        return image
        
    depths = [l['depth'] for l in layers]
    min_depth = min(depths)
    max_depth = max(depths)
    depth_range = max_depth - min_depth + 1e-5
    
    for layer in layers:
        rgba = layer['rgba']
        depth = layer['depth']
        
        # Normalized depth (0 = back, 1 = front)
        norm_depth = (depth - min_depth) / depth_range
        
        # Shift amounts
        shift_x = int(offset_x * norm_depth)
        shift_y = int(offset_y * norm_depth)
        
        # Create affine matrix for translation
        M = np.float32([[1, 0, shift_x], [0, 1, shift_y]])
        shifted_rgba = cv2.warpAffine(rgba, M, (W, H))
        
        rgb = shifted_rgba[..., :3]
        alpha = shifted_rgba[..., 3:4]
        
        # Composite
        result = rgb * alpha + result * (1 - alpha)
        alpha_acc = alpha + alpha_acc * (1 - alpha)
        
    return (result * 255).astype(np.uint8)


with gr.Blocks(title="Layered Representations") as app:
    gr.Markdown("# Layered Representations from a Single Image")
    gr.Markdown("Compare different architectures for decomposing an image into depth-ordered, semantically-grouped RGBA layers with intrinsic albedo/shading splits.")
    
    with gr.Row():
        with gr.Column(scale=1):
            input_image = gr.Image(type="pil", label="Input Image")
            model_dropdown = gr.Dropdown(choices=['Baseline A', 'Baseline B', 'Baseline C'], value='Baseline A', label="Architecture")
            process_btn = gr.Button("Decompose Image")
            
        with gr.Column(scale=2):
            gr.Markdown("### Layer Viewer (Back to Front)")
            layer_gallery = gr.Gallery(label="Layers", columns=3, height="auto")
            
    with gr.Row():
        with gr.Column():
            gr.Markdown("### Intrinsic Albedo")
            albedo_gallery = gr.Gallery(label="Albedo", columns=3, height="auto")
        with gr.Column():
            gr.Markdown("### Intrinsic Shading")
            shading_gallery = gr.Gallery(label="Shading", columns=3, height="auto")
            
    with gr.Row():
        with gr.Column():
            gr.Markdown("### 2.5D Parallax Simulator")
            x_slider = gr.Slider(minimum=-50, maximum=50, value=0, step=1, label="Horizontal Shift")
            y_slider = gr.Slider(minimum=-50, maximum=50, value=0, step=1, label="Vertical Shift")
            parallax_btn = gr.Button("Generate Parallax")
        with gr.Column():
            parallax_output = gr.Image(label="Parallax Result")
            
    process_btn.click(
        fn=process_image,
        inputs=[input_image, model_dropdown],
        outputs=[layer_gallery, albedo_gallery, shading_gallery]
    )
    
    parallax_btn.click(
        fn=generate_parallax,
        inputs=[input_image, model_dropdown, x_slider, y_slider],
        outputs=[parallax_output]
    )

if __name__ == "__main__":
    app.launch(server_name="0.0.0.0", server_port=7860)
