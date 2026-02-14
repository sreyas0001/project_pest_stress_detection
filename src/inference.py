
import os
import argparse
import torch
import numpy as np
import cv2
import matplotlib.pyplot as plt
from tqdm import tqdm
import rasterio
from rasterio.windows import Window

from model import UNet
from preprocessing import MultispectralImage

def predict_whole_image(model, image_path, patch_size=256, device='cpu'):
    model.eval()
    
    with rasterio.open(image_path) as src:
        H, W = src.height, src.width
        profile = src.profile
        
    # Create empty mask
    full_mask = np.zeros((H, W), dtype=np.uint8)
    
    # Iterate with windows
    # Note: If H or W is not divisible by patch_size, we need to handle edge cases.
    # Simple approach: pad or just process valid patches and leave edges.
    # Better approach: process with overlap or padding.
    # For this implementation, we will process valid patches and ignore the small strip at the edge 
    # (or better, process the last patch with overlap).
    
    # Let's simple stride
    stride = patch_size
    
    for y in tqdm(range(0, H - patch_size + 1, stride), desc="Predicting"):
        for x in range(0, W - patch_size + 1, stride):
            # Load patch
            img_obj = MultispectralImage(image_path, load_full=False)
            patch = img_obj.read_window(x, y, patch_size, patch_size)
            
            # Normalize
            patch = (patch - np.min(patch)) / (np.max(patch) - np.min(patch) + 1e-8)
            
            # To Tensor
            patch_tensor = torch.from_numpy(patch).float().unsqueeze(0).to(device) # (1, C, H, W)
            
            with torch.no_grad():
                output = model(patch_tensor)
                prob = torch.sigmoid(output)
                pred = (prob > 0.5).float().cpu().numpy().squeeze()
                
            full_mask[y:y+patch_size, x:x+patch_size] = pred.astype(np.uint8)
            
    return full_mask, profile

def visualize_results(image_path, mask, output_path, result_text):
    ms = MultispectralImage(image_path, load_full=True) # Load full for viz
    rgb = ms.get_rgb()
    ndvi = ms.calculate_ndvi()
    
    # Create figure
    fig, ax = plt.subplots(1, 3, figsize=(18, 6))
    
    ax[0].imshow(rgb)
    ax[0].set_title("Input RGB (False Color)")
    ax[0].axis('off')
    
    # NDVI Heatmap
    im1 = ax[1].imshow(ndvi, cmap='RdYlGn')
    ax[1].set_title("NDVI Index")
    ax[1].axis('off')
    plt.colorbar(im1, ax=ax[1], fraction=0.046, pad=0.04)
    
    # Prediction Overlay
    # Create a red overlay for stressed areas
    overlay = rgb.copy()
    # Mask is 0 or 1.
    # Where mask == 1 (Stressed), set Red channel to 255 (or high) and others low
    
    # Create an alpha blend
    # Red mask
    red_mask = np.zeros_like(rgb)
    red_mask[:, :, 0] = 255 # Red channel
    
    # Create alpha channel based on prediction
    alpha = (mask * 0.4)[:, :, None] # 40% opacity
    
    # Blend
    blended = (1 - alpha) * rgb + alpha * red_mask
    blended = np.clip(blended, 0, 255).astype(np.uint8)
    
    ax[2].imshow(blended)
    ax[2].set_title(f"Prediction (Stressed Areas)\n{result_text}")
    ax[2].axis('off')
    
    plt.tight_layout()
    plt.savefig(output_path)
    print(f"Visualization saved to {output_path}")

def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load Model
    model = UNet(n_channels=5, n_classes=1).to(device)
    if os.path.exists(args.model_path):
        model.load_state_dict(torch.load(args.model_path, map_location=device))
        print("Model loaded.")
    else:
        print("Model file not found. Initializing random weights (for testing only).")
    
    # Predict
    mask, profile = predict_whole_image(model, args.image_path, patch_size=256, device=device)
    
    # Analysis
    total_pixels = mask.size
    stressed_pixels = np.sum(mask)
    stressed_ratio = stressed_pixels / total_pixels
    
    print(f"Total Pixels: {total_pixels}")
    print(f"Stressed Pixels: {stressed_pixels}")
    print(f"Stressed Ratio: {stressed_ratio:.2%}")
    
    action = "NO ACTION REQUIRED"
    if stressed_ratio > args.threshold:
        action = "PEST CONTROL REQUIRED"
        
    print(f"Decision: {action}")
    
    # Visualize
    output_filename = os.path.splitext(os.path.basename(args.image_path))[0] + "_result.png"
    output_path = os.path.join(args.output_dir, output_filename)
    
    visualize_results(args.image_path, mask, output_path, result_text=f"{action} ({stressed_ratio:.1%})")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_path', type=str, required=True, help='Path to input multispectral image')
    parser.add_argument('--model_path', type=str, default='best_model.pth', help='Path to trained model')
    parser.add_argument('--output_dir', type=str, default='.', help='Directory to save results')
    parser.add_argument('--threshold', type=float, default=0.10, help='Threshold for pest control application (e.g., 0.10 for 10%)')
    
    args = parser.parse_args()
    main(args)
