"""
Enhanced dataset generator for astronomical image sharpening
Optimized for pre-stretched Hubble images
"""

import os
import glob
import numpy as np
import cv2
from PIL import Image
import tifffile
import argparse
from tqdm import tqdm
import matplotlib.pyplot as plt
from skimage import filters, measure, morphology
from scipy import ndimage, stats
import random
import uuid
import json
import sys
import warnings
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from skimage import feature
from skimage import measure
from PIL import Image
from scipy import special
from concurrent.futures import ThreadPoolExecutor, as_completed

# Use non-interactive matplotlib backend to avoid thread issues
import matplotlib
matplotlib.use('Agg')  # Must be before any matplotlib.pyplot import

# Disable PIL's show functionality to prevent Tkinter errors
from PIL import Image
Image.Image.show = lambda *args, **kwargs: None

warnings.filterwarnings("ignore", category=UserWarning)

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Generate training dataset for astrophotography sharpening')
    parser.add_argument('--input-dir', type=str, required=True, 
                       help='Directory containing input TIFF images')
    parser.add_argument('--output-dir', type=str, required=True,
                       help='Base output directory')
    parser.add_argument('--tile-size', type=int, default=256,
                       help='Size of output tiles (default: 256)')
    parser.add_argument('--min-extraction-size', type=int, default=512,
                       help='Minimum size of extraction tiles before resizing (default: 512)')
    parser.add_argument('--max-extraction-size', type=int, default=2048,
                       help='Maximum size of extraction tiles before resizing (default: 2048)')
    parser.add_argument('--overlap', type=int, default=64,
                       help='Overlap between tiles (default: 64)')
    parser.add_argument('--min-brightness', type=float, default=0.02,
                       help='Minimum brightness threshold for tiles (0-1)')
    parser.add_argument('--min-structure', type=float, default=0.01,
                       help='Minimum structure threshold for tiles (0-1)')
    parser.add_argument('--augmentations', type=int, default=3,
                       help='Number of augmentations per tile (default: 3)')
    parser.add_argument('--debug', action='store_true',
                       help='Save debug visualizations')
    parser.add_argument('--max-tiles', type=int, default=None,
                       help='Maximum number of tiles to extract per image (for testing)')
    parser.add_argument('--min-contrast', type=float, default=0.08,
                       help='Minimum contrast for tile acceptance')
    parser.add_argument('--estimate-only', action='store_true',
                       help='Only estimate the number of tiles without creating them')
    parser.add_argument('--prefer-astro-features', action='store_true', default=True,
                    help='Prioritize astronomical features (galaxies and nebulae) over stars')
    parser.add_argument('--tiles-only', action='store_true',
                       help='Only select tiles without processing them')
    parser.add_argument('--preserve-noise', action='store_true', default=True,  # Changed default
                       help='Preserve noise patterns in blurred versions')
    parser.add_argument('--noise-preservation', type=float, default=0.8,  # Increased from 0.6
                      help='Noise preservation factor (0-1, default: 0.8)')
    parser.add_argument('--edge-aware-blur', action='store_true', default=False,
                      help='Use edge-aware blurring to better preserve fine details')
    parser.add_argument('--denoise-strength', type=float, default=0.4,  # Changed from 0.2
                    help='Strength of denoising applied to ground truth (0-1, default: 0.4)')
    parser.add_argument('--moffat-beta', type=float, default=2.5,  # Changed from 4.0 to 2.5 (more realistic for Hubble)
                      help='Beta parameter for Moffat PSF (2.5 typical for Hubble data)')
    parser.add_argument('--auto-denoise-only', action='store_true', default=True,
                      help='Only apply denoising when excessive noise is detected')
    parser.add_argument('--diverse-blur', action='store_true', default=False,
                      help='Use diverse blur types beyond just Moffat PSF')
    return parser.parse_args()

def create_directory_structure(base_dir):
    """Create the directory structure for the dataset"""
    # Main directories
    os.makedirs(base_dir, exist_ok=True)
    
    # Create subdirectories for different blur levels, now with 4 blur levels
    dirs = {
        'ground_truth': os.path.join(base_dir, 'Ground Truth'),
        'blur_1': os.path.join(base_dir, 'Blur 1'),
        'blur_2': os.path.join(base_dir, 'Blur 2'),
        'blur_3': os.path.join(base_dir, 'Blur 3'),
        'blur_4': os.path.join(base_dir, 'Blur 4'),
        'background': os.path.join(base_dir, 'Background'),
        'debug': os.path.join(base_dir, 'Debug')
    }
    
    for dir_path in dirs.values():
        os.makedirs(dir_path, exist_ok=True)
    
    return dirs

def load_tiff_image(file_path):
    """
    Load a 32-bit floating point TIFF image safely with proper handling
    Returns the image normalized to [0, 1] range
    """
    try:
        # Use tifffile for proper handling of 32-bit float TIFFs
        img = tifffile.imread(file_path)
        
        # Check if image loaded correctly
        if img is None:
            raise ValueError(f"Failed to load image from {file_path}")
            
        # Convert to float32 if not already
        img = img.astype(np.float32)
        
        # Store original range
        orig_min = np.min(img)
        orig_max = np.max(img)
        orig_range = (orig_min, orig_max)
        
        # Handle different dimensions (ensure 3 dimensions with channel last)
        if len(img.shape) == 2:  # Single channel grayscale
            img = img[..., np.newaxis]
        elif len(img.shape) == 3 and img.shape[0] == 3:  # Channel-first format
            img = np.transpose(img, (1, 2, 0))
        
        # Handle NaN and Infinity values
        img = np.nan_to_num(img, nan=0.0, posinf=1.0, neginf=0.0)
            
        # Normalize to [0, 1] range - gentle normalization preserving data characteristics
        normalized = np.zeros_like(img)
        for c in range(img.shape[2]):
            # Simple min-max normalization
            min_val = np.min(img[:,:,c])
            max_val = np.max(img[:,:,c])
            
            if max_val > min_val:
                normalized[:,:,c] = (img[:,:,c] - min_val) / (max_val - min_val)
            else:
                normalized[:,:,c] = np.zeros_like(img[:,:,c])
            
        return normalized, orig_range
        
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return None, (0, 1)

def detect_astronomical_features(image, min_feature_size=250):
    """
    Improved astronomical feature detection - less aggressive, more accurate
    Better handling of faint galaxies and nebulae without over-grabbing
    """
    # Convert to grayscale if needed
    if len(image.shape) > 2 and image.shape[2] > 1:
        gray = np.mean(image, axis=2)
    else:
        gray = np.squeeze(image)
    
    # Apply moderate blur to eliminate noise but preserve structure
    smoothed = cv2.GaussianBlur(gray, (5, 5), 1.5)  # Reduced sigma from 2.0
    
    # Use more conservative thresholding
    high_thresh = np.percentile(smoothed, 92)  # Increased from 88 (more selective)
    high_mask = smoothed > high_thresh
    
    # Medium threshold for connected structures
    med_thresh = np.percentile(smoothed, 82)  # Increased from 75 (more selective)
    med_mask = smoothed > med_thresh
    
    # Lower threshold for very faint extensions
    low_thresh = np.percentile(smoothed, 70)  # Reduced from 75 (less aggressive)
    low_mask = smoothed > low_thresh
    
    # Clean up the high mask - smaller kernels for precision
    kernel_small = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    kernel_medium = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    kernel_large = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))  # Reduced from (9, 9)
    
    high_cleaned = cv2.morphologyEx(high_mask.astype(np.uint8), cv2.MORPH_OPEN, kernel_small)
    high_cleaned = cv2.morphologyEx(high_cleaned, cv2.MORPH_CLOSE, kernel_medium)
    
    # More conservative small object removal
    if image.shape[0] * image.shape[1] > 1000000:  # 1 megapixel threshold
        num_labels, labels = cv2.connectedComponents(high_cleaned)
        sizes = np.bincount(labels.flatten())
        mask_sizes = sizes > min_feature_size * 1.5  # Increased threshold
        mask_sizes[0] = 0  # Background should be 0
        high_cleaned = mask_sizes[labels]
        high_cleaned = high_cleaned.astype(np.uint8)
    
    # Conservative dilation for seeds
    seeds = cv2.dilate(high_cleaned, kernel_medium)  # Reduced from kernel_large
    
    # Multi-level connection: high -> medium -> low (more controlled)
    # First connect high to medium
    med_connected = (seeds > 0) & med_mask
    med_connected = cv2.morphologyEx(med_connected.astype(np.uint8), cv2.MORPH_CLOSE, kernel_small)
    
    # Then extend to low areas, but only from medium areas
    med_seeds = cv2.dilate(med_connected, kernel_medium)
    low_connected = (med_seeds > 0) & low_mask
    
    # Combine all levels
    final_connected = np.logical_or(high_cleaned > 0, 
                                  np.logical_or(med_connected > 0, low_connected > 0))
    
    # Clean up with conservative morphology
    num_labels, labels = cv2.connectedComponents(final_connected.astype(np.uint8))
    
    if num_labels > 1:  # Remember, label 0 is background
        sizes = np.bincount(labels.flatten())
        
        # Keep fewer but larger components
        max_components = min(4, num_labels-1)  # Reduced from 6
        
        valid_indices = np.arange(1, len(sizes))  # Skip background
        valid_sizes = sizes[1:]  # Skip background
        
        if len(valid_sizes) > 0:
            largest_indices = valid_indices[np.argsort(valid_sizes)[-max_components:]]
            
            # Create final mask with only the largest components
            final_mask = np.zeros_like(final_connected, dtype=bool)
            for idx in largest_indices:
                component = labels == idx
                # Much more conservative dilation
                kernel_expand = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))  # Reduced from (15, 15)
                dilated = cv2.dilate(component.astype(np.uint8), kernel_expand)
                final_mask = np.logical_or(final_mask, dilated > 0)
            
            # Final conservative expansion
            kernel_final = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (13, 13))  # Much smaller than before
            final_mask = cv2.dilate(final_mask.astype(np.uint8), kernel_final)
            return final_mask > 0
        
    # If no significant components found
    return final_connected

def extract_background_samples(image, astro_mask, num_samples=20, patch_size=256):
    """
    Extract background samples with slightly relaxed filtering to allow more samples
    while still ensuring reasonably dark backgrounds
    """
    # Get dimensions
    height, width = image.shape[:2]
    
    # Ensure we have a valid mask
    if astro_mask is None:
        astro_mask = detect_astronomical_features(image)
    
    # Apply dilation to ensure we're far from any structures, but slightly less extreme
    dilated_mask = morphology.binary_dilation(astro_mask, morphology.disk(25))  # Reduced from 30
    
    # Get the background regions (inverse of the dilated mask)
    background_mask = ~dilated_mask
    
    # Skip if not enough background regions
    if np.sum(background_mask) < patch_size * patch_size * 3:
        print("Warning: Not enough background found after dilation.")
        # Try with a smaller dilation as fallback
        dilated_mask = morphology.binary_dilation(astro_mask, morphology.disk(18))  # Reduced from 20
        background_mask = ~dilated_mask
        if np.sum(background_mask) < patch_size * patch_size:
            print("Not enough background regions found. Returning empty list.")
            return []
    
    # Extract random patches from background
    background_patches = []
    attempts = 0
    max_attempts = num_samples * 60  # Increased from 40 to accommodate more samples
    last_report = 0
    report_interval = 50  # Update every 50 attempts

    print(f"Searching for {num_samples} background samples (max {max_attempts} attempts)...")
    while len(background_patches) < num_samples and attempts < max_attempts:
        attempts += 1
        
        # Print progress update at regular intervals
        if attempts - last_report >= report_interval:
            print(f"  Background search: {attempts}/{max_attempts} attempts, found {len(background_patches)}/{num_samples} samples")
            last_report = attempts
        
        # Find valid background locations
        valid_y, valid_x = np.where(background_mask)
        if len(valid_y) < patch_size:  # Skip if not enough background
            break
            
        # Select random position
        idx = random.randint(0, len(valid_y)-1)
        y, x = valid_y[idx], valid_x[idx]
        
        # Extract patch ensuring we stay within bounds
        if y + patch_size > height or x + patch_size > width:
            continue
        
        # Require background purity - slightly looser
        patch_mask = background_mask[y:y+patch_size, x:x+patch_size]
        if np.mean(patch_mask) < 0.92:  # Reduced from 0.95 to 0.92
            continue
            
        patch = image[y:y+patch_size, x:x+patch_size].copy()
            
        # Checks for reasonably pure background
        # Convert to grayscale for analysis
        gray = np.mean(patch, axis=2) if len(patch.shape) > 2 else patch
        
        # BRIGHTNESS CHECK: Find a balance between v1 and v2
        if np.mean(gray) > 0.10:  # Slightly more strict than v2's 0.12, but more permissive than v1's 0.08
            continue
            
        # Check max brightness - balanced approach  
        if np.max(gray) > 0.22:  # Between v2's 0.25 and v1's 0.2
            continue
            
        # Check for any structure using edge detection - balanced approach
        edge_mag = filters.sobel(gray)
        if np.max(edge_mag) > 0.55:  # Between v2's 0.6 and v1's 0.5
            continue

        # More balanced check on standard deviation
        std_val = np.std(gray)
        if std_val > 0.045:  # Between v2's 0.05 and v1's 0.04
            continue
        
        # Check for color neutrality - looser criteria
        if len(patch.shape) > 2 and patch.shape[2] >= 3:
            # Calculate difference between channels
            r, g, b = patch[:,:,0], patch[:,:,1], patch[:,:,2]
            r_mean, g_mean, b_mean = np.mean(r), np.mean(g), np.mean(b)
                    
            # Allow more color variation
            channel_deviation = max(abs(r_mean - g_mean), abs(r_mean - b_mean), abs(g_mean - b_mean))
            if channel_deviation > 0.04:  # Increased from 0.03 to 0.04
                continue
            
            # Allow slightly brighter backgrounds
            if r_mean > 0.15 or g_mean > 0.15 or b_mean > 0.15:  # Increased from 0.1 to 0.15
                continue
        
        # Check for uniformity of noise - slightly looser
        h_mid, w_mid = patch_size//2, patch_size//2
        q1 = gray[:h_mid, :w_mid]
        q2 = gray[:h_mid, w_mid:]
        q3 = gray[h_mid:, :w_mid]
        q4 = gray[h_mid:, w_mid:]
        
        quadrant_means = [np.mean(q) for q in [q1, q2, q3, q4]]
        overall_mean = np.mean(gray)
        
        # Allow slightly more variation between quadrants
        if any(abs(qm - overall_mean) / (overall_mean + 1e-10) > 0.04 for qm in quadrant_means):  # Increased from 0.03 to 0.04
            continue
            
        background_patches.append({
            'patch': patch,
            'position': (x, y),
            'size': (patch_size, patch_size)
        })
    
    print(f"Found {len(background_patches)} background samples after {attempts} attempts")
    
    # Final check - with looser threshold
    if background_patches:
        avg_brightness = np.mean([np.mean(p['patch']) for p in background_patches])
        if avg_brightness > 0.08:  # Increased from 0.05 to 0.08
            print(f"WARNING: Background samples may not be dark enough (avg: {avg_brightness:.4f})")
    
    return background_patches

def get_image_structure_score(image, astro_mask=None):
    """
    Calculate an image structure score based on edge content and local variance
    Higher values indicate more structure (astronomical object details) in the image
    """
    # Convert to grayscale if needed
    if len(image.shape) > 2 and image.shape[2] > 1:
        gray = np.mean(image, axis=2)
    else:
        gray = np.squeeze(image)
    
    # Calculate edge magnitude using Sobel
    sobel_x = filters.sobel_h(gray)
    sobel_y = filters.sobel_v(gray)
    edge_mag = np.sqrt(sobel_x**2 + sobel_y**2)
    
    # Calculate local variance (window size based on image size)
    window_size = max(3, min(image.shape[0] // 20, 11))
    if window_size % 2 == 0:
        window_size += 1  # Ensure odd window size
    local_var = ndimage.generic_filter(gray, np.var, size=window_size)
    
    # Basic structure score
    structure_score = (np.mean(edge_mag) * 0.6 + np.mean(local_var) * 0.4)
    
    # If astro mask is provided, weight structure by mask
    if astro_mask is not None and np.any(astro_mask):
        astro_structure = np.mean(edge_mag[astro_mask]) * 0.6 + np.mean(local_var[astro_mask]) * 0.4
        structure_score = structure_score * 0.3 + astro_structure * 0.7
    
    return structure_score

def calculate_tile_metrics(tile, astro_mask=None):
    """
    Enhanced tile metrics calculation that values subtle structures and faint features
    Optimized for performance while preserving quality assessment
    """
    # Convert to grayscale if needed
    if len(tile.shape) > 2 and tile.shape[2] > 1:
        gray = np.mean(tile, axis=2)
    else:
        gray = np.squeeze(tile)
    
    # Handle NaN values
    gray = np.nan_to_num(gray, nan=0.0)
    
    # Basic statistics - faster implementations
    mean_val = np.mean(gray)
    std_val = np.std(gray)
    min_val = np.min(gray)
    max_val = np.max(gray)
    contrast = max_val - min_val
    
    # Quick return if contrast is too low - avoids expensive calculations
    if contrast < 0.03:  # Lowered threshold
        return {
            'mean': float(mean_val),
            'std': float(std_val),
            'contrast': float(contrast),
            'edge_mean': 0.0,
            'texture_score': 0.0,
            'entropy': 0.0,
            'dark_ratio': 0.0,
            'structure_score': 0.0,
            'astro_coverage': 0.0,
            'bg_ratio': 1.0,
            'snr': 0.0,
            'faint_structure_score': 0.0  # NEW
        }
    
    # Edge content - enhanced to detect subtle edges using faster cv2 methods
    edge_mag = cv2.Sobel(gray, cv2.CV_32F, 1, 1, ksize=3)
    edge_mag = np.abs(edge_mag)
    edge_mean = np.mean(edge_mag)
    
    # Texture analysis - useful for dust lanes but optimized
    # Use 5x5 blocks for variance calculation - more efficient than ndimage.generic_filter
    local_var = cv2.blur(gray**2, (5, 5)) - cv2.blur(gray, (5, 5))**2
    texture_score = np.mean(local_var)
    
    # Entropy - measure of information content - use fewer bins for speed
    nbins = 128  # Reduced from 256
    hist, _ = np.histogram(gray, nbins, (0, 1))
    hist = hist / np.sum(hist)
    hist_valid = hist > 0
    entropy = -np.sum(hist[hist_valid] * np.log2(hist[hist_valid])) if np.any(hist_valid) else 0
    
    # NEW: Faint feature detection
    # Identify faint structures that are above background but below bright features
    percentile_95 = np.percentile(gray, 95)
    percentile_80 = np.percentile(gray, 80)
    percentile_20 = np.percentile(gray, 20)  # Background estimate
    
    # Faint features are between 80th and 95th percentile
    faint_mask = (gray > percentile_80) & (gray < percentile_95) & (gray > percentile_20 * 2)
    faint_structure_score = np.sum(faint_mask) / gray.size
    
    # Also check for faint extended structures using gradient
    faint_edges = edge_mag[faint_mask].mean() if np.any(faint_mask) else 0
    faint_structure_score = faint_structure_score * 0.5 + faint_edges * 10  # Weight edge content
    
    # Calculate astronomical object coverage if mask is provided
    astro_coverage = 0
    if astro_mask is not None:
        astro_coverage = np.mean(astro_mask)
    
    # Calculate dark feature ratio (for dust lanes)
    dark_threshold = min_val + (max_val - min_val) * 0.3
    dark_ratio = np.mean((gray < dark_threshold) & (edge_mag > 0.02))
    
    # Calculate background ratio
    bg_threshold = min_val + (max_val - min_val) * 0.15
    bg_ratio = np.mean(gray < bg_threshold)
    
    # Combine into structure score with more weight to texture, entropy, and faint features
    structure_score = (
        edge_mean * 0.25 + 
        std_val * 0.15 + 
        texture_score * 0.20 + 
        entropy / 8.0 * 0.15 +  # Normalize entropy
        dark_ratio * 0.10 +
        faint_structure_score * 0.15  # NEW: Weight faint features
    )
    
    # Boost structure score for low contrast features (important for faint galaxies)
    if contrast < 0.15:  # For low contrast features
        structure_score *= 1.3  # 30% boost instead of 50%
    
    # Additional boost for tiles with significant faint structure
    if faint_structure_score > 0.1:
        structure_score *= 1.2
    
    return {
        'mean': float(mean_val),
        'std': float(std_val),
        'contrast': float(contrast),
        'edge_mean': float(edge_mean),
        'texture_score': float(texture_score),
        'entropy': float(entropy),
        'dark_ratio': float(dark_ratio),
        'structure_score': float(structure_score),
        'astro_coverage': float(astro_coverage),
        'bg_ratio': float(bg_ratio),
        'snr': float(edge_mean / 0.001),
        'faint_structure_score': float(faint_structure_score)  # NEW
    }

def classify_tile_type(metrics):
    """
    Classify a tile based on its metrics to ensure diverse feature representation
    Returns one of: 'bright_core', 'spiral_arm', 'dust_lane', 'diffuse_edge', 'low_contrast'
    """
    if metrics['mean'] > 0.3 and metrics['contrast'] > 0.4:  # More permissive
        return 'bright_core'
    elif metrics['dark_ratio'] > 0.1:  # More permissive
        return 'dust_lane'
    elif metrics['contrast'] < 0.2:  # More permissive
        return 'low_contrast'
    elif metrics['edge_mean'] > 0.04 and metrics['texture_score'] > 0.015:  # More permissive
        return 'spiral_arm'
    else:
        return 'diffuse_edge'

def estimate_noise_level(image):
    """
    Estimate noise level in an image using median absolute deviation
    Returns an estimate of the noise standard deviation
    """
    # Calculate image gradients
    gx = np.diff(image, axis=1)
    gy = np.diff(image, axis=0)
    
    # Pad to restore original shape
    gx = np.pad(gx, ((0, 0), (0, 1)), mode='constant')
    gy = np.pad(gy, ((0, 1), (0, 0)), mode='constant')
    
    # Combine gradients
    grad = np.sqrt(gx**2 + gy**2)
    
    # Use median absolute deviation as a robust noise estimator
    mad = np.median(np.abs(grad - np.median(grad)))
    
    # Convert MAD to standard deviation (assuming normal distribution)
    noise_std = mad * 1.4826
    
    return noise_std

def add_realistic_noise(image, noise_level=0.01, poisson_factor=0.5):
    """
    Add realistic astronomical noise to an image:
    - Gaussian noise (background, readout)
    - Poisson noise (photon shot noise)
    - Hot pixels (cosmic rays)
    """
    # Make a copy to avoid modifying the original
    noisy_image = image.copy()
    
    # Add Gaussian noise (background, readout)
    gaussian_noise = np.random.normal(0, noise_level, image.shape)
    
    # Add Poisson noise (signal-dependent shot noise)
    if poisson_factor > 0:
        # Scale image to higher values for realistic Poisson
        scaled = image * 100
        poisson_noise = np.random.poisson(scaled + 1.0) / 100.0 - scaled / 100.0
        poisson_noise *= poisson_factor
    else:
        poisson_noise = 0
    
    # Add hot pixels (cosmic rays) - very sparse
    hot_pixels = np.zeros_like(image)
    if len(image.shape) > 2:
        for c in range(image.shape[2]):
            # Add random hot pixels (about 0.01% of pixels)
            num_hot = max(1, int(0.0001 * image.shape[0] * image.shape[1]))
            hot_y = np.random.randint(0, image.shape[0], num_hot)
            hot_x = np.random.randint(0, image.shape[1], num_hot)
            hot_val = np.random.uniform(0.7, 1.0, num_hot)
            
            for i in range(num_hot):
                # Add a small cluster around each hot pixel
                size = np.random.randint(1, 4)
                y, x = hot_y[i], hot_x[i]
                if y < image.shape[0]-size and x < image.shape[1]-size:
                    hot_pixels[y:y+size, x:x+size, c] = hot_val[i]
    else:
        # For grayscale images
        num_hot = max(1, int(0.0001 * image.shape[0] * image.shape[1]))
        hot_y = np.random.randint(0, image.shape[0], num_hot)
        hot_x = np.random.randint(0, image.shape[1], num_hot)
        hot_val = np.random.uniform(0.7, 1.0, num_hot)
        
        for i in range(num_hot):
            size = np.random.randint(1, 4)
            y, x = hot_y[i], hot_x[i]
            if y < image.shape[0]-size and x < image.shape[1]-size:
                hot_pixels[y:y+size, x:x+size] = hot_val[i]
    
    # Combine all noise sources
    noisy_image = np.clip(noisy_image + gaussian_noise + poisson_noise + hot_pixels, 0, 1)
    
    return noisy_image

def add_photon_noise(image, preserve_snr=True, base_level=0.001):
    """
    Add realistic photon (Poisson) noise that scales with signal
    Preserves SNR characteristics of astronomical images
    """
    # Make a copy
    noisy = image.copy()
    
    # Scale to physical units (arbitrary but consistent)
    # Higher values = better Poisson statistics
    scale_factor = 10000
    scaled = image * scale_factor
    
    # Add Poisson noise
    # For each pixel, the noise scales with sqrt(signal)
    if len(image.shape) > 2:
        for c in range(image.shape[2]):
            # Ensure non-negative values for Poisson
            scaled_channel = np.maximum(scaled[:,:,c], 0) + 1.0
            noisy[:,:,c] = np.random.poisson(scaled_channel) / scale_factor
    else:
        scaled = np.maximum(scaled, 0) + 1.0
        noisy = np.random.poisson(scaled) / scale_factor
    
    if preserve_snr:
        # Preserve the overall SNR by scaling noise
        original_std = np.std(image)
        noise_component = noisy - image
        noise_std = np.std(noise_component)
        
        if noise_std > 0 and original_std > 0:
            # Target SNR based on typical Hubble data
            target_snr = 30  # Adjust based on your data
            scale = original_std / (noise_std * target_snr)
            noisy = image + noise_component * scale
    
    # Add small amount of Gaussian noise for read noise
    gaussian = np.random.normal(0, base_level, image.shape)
    noisy = noisy + gaussian
    
    return np.clip(noisy, 0, 1)


def edge_aware_blur(image, sigma):
    """Apply edge-aware blurring to preserve fine details"""
    if len(image.shape) > 2:
        result = np.zeros_like(image)
        for c in range(image.shape[2]):
            # Use bilateral filter for edge-aware smoothing
            # Parameters: image, diameter, sigmaColor, sigmaSpace
            result[:,:,c] = cv2.bilateralFilter(
                image[:,:,c].astype(np.float32), 
                0,  # Auto diameter
                sigma*0.3,  # Color sigma - controls how much similar colors are mixed
                sigma  # Space sigma - standard deviation of Gaussian for spatial distance
            )
        return result
    else:
        return cv2.bilateralFilter(image.astype(np.float32), 0, sigma*0.3, sigma)
    
def detect_excessive_noise(image, threshold=0.15):
    """
    Detect if an image has excessive noise that would benefit from light denoising
    Returns True if noise is excessive, False otherwise
    """
    # Convert to grayscale for analysis
    if len(image.shape) > 2 and image.shape[2] > 1:
        gray = np.mean(image, axis=2)
    else:
        gray = np.squeeze(image)
    
    # Estimate noise using median absolute deviation in high-frequency
    # Calculate gradients
    gx = np.diff(gray, axis=1)
    gy = np.diff(gray, axis=0)
    
    # Combine gradients
    grad = np.sqrt(np.pad(gx, ((0,0), (0,1)), mode='constant')**2 + 
                   np.pad(gy, ((0,1), (0,0)), mode='constant')**2)
    
    # Use MAD as robust noise estimator
    mad = np.median(np.abs(grad - np.median(grad)))
    noise_estimate = mad * 1.4826
    
    # Calculate signal strength
    signal_std = np.std(cv2.GaussianBlur(gray, (0, 0), sigmaX=2.0))
    
    # Calculate noise-to-signal ratio
    if signal_std > 0:
        noise_ratio = noise_estimate / signal_std
        return noise_ratio > threshold
    
    return False

def denoise_image(image, strength=0.3):
    """
    Apply VERY gentle denoising suitable for astronomical images
    Only removes high-frequency noise while preserving ALL structure
    """
    if strength <= 0:
        return image
    
    # Much more conservative for astronomical data
    # Maximum strength capped at 0.3 to preserve faint features
    strength = min(strength, 0.3)
        
    # Create a copy to avoid modifying the original
    result = image.copy()
    
    # Use bilateral filter for edge-preserving denoising
    # This preserves structure better than Gaussian
    sigma_color = 0.05 + (strength * 0.1)  # Very conservative color sigma
    sigma_space = 0.5 + (strength * 1.0)   # Small spatial sigma
    
    if len(image.shape) > 2:
        # Process each channel with bilateral filter
        for c in range(image.shape[2]):
            result[:,:,c] = cv2.bilateralFilter(
                image[:,:,c].astype(np.float32), 
                0,  # Auto diameter
                sigma_color,  # Preserve similar intensities
                sigma_space   # Spatial extent
            )
    else:
        # For grayscale images
        result = cv2.bilateralFilter(
            image.astype(np.float32), 0, sigma_color, sigma_space)
    
    # Very conservative blending - preserve most of original
    alpha = strength * 0.3  # Maximum 30% of denoised version
    blended = (1-alpha) * image + alpha * result
    
    return blended

def check_color_preservation(original, blurred):
    """
    Check if color balance is preserved between original and blurred version
    Returns True if balanced, False if not, and logs the discrepancy
    """
    if len(original.shape) < 3 or original.shape[2] < 3:
        return True  # No color preservation needed for grayscale
    
    # Get mean values for each channel
    orig_means = [np.mean(original[:,:,c]) for c in range(original.shape[2])]
    blur_means = [np.mean(blurred[:,:,c]) for c in range(blurred.shape[2])]
    
    # Calculate channel ratios for original
    if orig_means[0] > 0 and orig_means[1] > 0 and orig_means[2] > 0:
        orig_ratios = [orig_means[0]/orig_means[1], orig_means[0]/orig_means[2], orig_means[1]/orig_means[2]]
        blur_ratios = [blur_means[0]/blur_means[1], blur_means[0]/blur_means[2], blur_means[1]/blur_means[2]]
        
        # Calculate percent differences in ratios
        ratio_diffs = [abs(orig_ratios[i] - blur_ratios[i])/orig_ratios[i] for i in range(3)]
        max_diff = max(ratio_diffs) * 100  # Convert to percentage
        
        # If any ratio differs by more than 2%, flag it
        if max_diff > 2.0:
            print(f"Warning: Color balance shift detected: {max_diff:.2f}% max channel ratio difference")
            return False
            
    return True

def select_tiles_intelligently(image, astro_mask=None, tile_size=256, overlap=64, 
                             min_extraction_size=512, max_extraction_size=2048, max_tiles=None, 
                             min_contrast=0.06, min_structure=0.008, min_brightness=0.02):
    """
    Extract and select high-quality tiles focusing on astronomical features
    Uses an intelligent selection algorithm with progress reporting
    Optimized for performance and to produce more tiles
    """
    # Get image dimensions
    height, width = image.shape[:2]
    
    # Define extraction tile sizes ensuring square aspect ratio - REDUCED OPTIONS
    tile_sizes = []
    
    # Use fewer tile sizes for better performance
    for size in [512, 768, 1024]:  # Reduced set of sizes
        if min_extraction_size <= size <= max_extraction_size and size <= min(height, width):
            tile_sizes.append((size, size))
    
    # Sort sizes and remove duplicates
    tile_sizes = sorted(list(set(tile_sizes)))
    print(f"Using extraction tile sizes: {[s[0] for s in tile_sizes]}")
    
    # Track contrast distribution for diversity
    contrast_bins = {
        "low": (min_contrast*0.7, 0.2),  # Reduced lower threshold by 30%
        "medium": (0.2, 0.4),
        "high": (0.4, 1.0)
    }
    
    # NEW: Add brightness bins for better distribution
    brightness_bins = {
        "very_dark": (0.0, 0.1),
        "dark": (0.1, 0.2),
        "medium": (0.2, 0.4),
        "bright": (0.4, 1.0)
    }
    
    # NEW: Track feature types for balanced dataset
    feature_types = {
        'bright_core': 0,
        'spiral_arm': 0,
        'dust_lane': 0,
        'diffuse_edge': 0,
        'low_contrast': 0
    }
    
    # NEW: Track spatial regions
    spatial_regions = {
        'core': 0,
        'outskirts': 0
    }
    
    # Generate all possible tile positions and evaluate them
    all_tiles = []
    
    # First generate tiles that specifically target astronomical features
    if astro_mask is not None and np.any(astro_mask):
        print("Finding feature-centered tiles...")
        # Find connected components in the astro mask - FIXED METHOD
        ret, labeled_mask = cv2.connectedComponents(astro_mask.astype(np.uint8))
        num_features = ret  # ret is the number of labels including background
        print(f"Found {num_features-1} astronomical features")  # Subtract 1 for background
        
        # Use a more efficient approach for large images
        if max(height, width) > 3000:
            # For very large images, use a density-based approach
            # Create a downsampled density map
            scale = min(1.0, 500 / max(height, width))
            small_h, small_w = int(height * scale), int(width * scale)
            density = cv2.resize(astro_mask.astype(np.uint8), (small_w, small_h), 
                               interpolation=cv2.INTER_AREA)
            
            # Find hotspots in the density map using non-max suppression
            kernel_size = max(3, int(50 * scale))
            max_filtered = cv2.dilate(density, np.ones((kernel_size, kernel_size), np.uint8))
            maxima = (density == max_filtered) & (density > 0)
            maxima_coords = np.where(maxima)
            
            # Convert back to original image coordinates but limit the number of centers
            feature_centers = [(int(x / scale), int(y / scale)) 
                             for y, x in zip(maxima_coords[0], maxima_coords[1])]
                             
            # Limit to maximum 800 feature centers for performance
            if len(feature_centers) > 800:  # Increased from 500
                print(f"Limiting from {len(feature_centers)} to 800 feature centers for performance")
                # Use random sampling to maintain distribution
                feature_centers = random.sample(feature_centers, 800)
            else:
                print(f"Found {len(feature_centers)} feature centers for large image")
        else:
            # For each significant feature, compute center of mass
            feature_centers = []
            for feature_idx in range(1, num_features):  # Skip background (0)
                # Get coordinates of this feature
                feature_coords = np.where(labeled_mask == feature_idx)
                
                # Decrease the minimum area requirement to capture more subtle features
                if len(feature_coords[0]) < 50:  # Reduced from 75 to 50
                    continue
                    
                # Find the center of mass of this feature
                center_y = int(np.mean(feature_coords[0]))
                center_x = int(np.mean(feature_coords[1]))
                feature_centers.append((center_x, center_y))
            
            if len(feature_centers) > 800:  # Increased from 500
                    print(f"Limiting from {len(feature_centers)} to 800 feature centers for performance")
                    feature_centers = random.sample(feature_centers, 800)
        
        # Process each feature center with parallel processing
        print(f"Evaluating tiles centered on {len(feature_centers)} features...")
        # For efficiency, process every Nth feature if there are too many
        sampling_rate = max(1, len(feature_centers) // 100)  # Process at most ~100 features
        sampled_centers = feature_centers[::sampling_rate]

        def process_feature_center(center_data):
            center_x, center_y = center_data
            
            # Use a random tile size to get different zoom levels
            size_idx = random.randint(0, len(tile_sizes)-1)
            tile_width, tile_height = tile_sizes[size_idx]
                
            # Calculate top-left corner to center the tile on the feature
            y = max(0, center_y - tile_height // 2)
            x = max(0, center_x - tile_width // 2)
            
            # Adjust if the tile would go beyond image boundaries
            if y + tile_height > height:
                y = height - tile_height
            if x + tile_width > width:
                x = width - tile_width
            
            # Extract tile from original image
            try:
                orig_tile = image[y:y+tile_height, x:x+tile_width].copy()
                
                # Quick contrast check before full metric calculation
                if len(orig_tile.shape) > 2 and orig_tile.shape[2] > 1:
                    gray = np.mean(orig_tile, axis=2)
                else:
                    gray = np.squeeze(orig_tile)
                    
                min_val, max_val = np.min(gray), np.max(gray)
                contrast = max_val - min_val
                
                # Skip low contrast tiles early with a more permissive threshold
                if contrast < min_contrast * 0.5:  # More permissive early check
                    return None
                
                # Get astro mask for this tile if available
                tile_astro_mask = None
                if astro_mask is not None:
                    tile_astro_mask = astro_mask[y:y+tile_height, x:x+tile_width].copy()
                
                # Calculate metrics for quality assessment
                metrics = calculate_tile_metrics(orig_tile, tile_astro_mask)
                
                # Skip extremely dark areas that should be background
                if metrics['mean'] < 0.035 and metrics['edge_mean'] < 0.03:  # Reject very dark areas with minimal edges
                    return None
                
                # NEW: Classify the tile feature type
                feature_type = classify_tile_type(metrics)
                
                # NEW: Determine spatial region
                spatial_region = 'outskirts'
                if astro_mask is not None:
                    # Find astronomical object center approximation
                    if np.any(astro_mask):
                        center_y_astro, center_x_astro = ndimage.center_of_mass(astro_mask)
                        # Calculate normalized distance from center
                        tile_center_y = y + tile_height/2
                        tile_center_x = x + tile_width/2
                        dist_from_center = np.sqrt((tile_center_x - center_x_astro)**2 + (tile_center_y - center_y_astro)**2)
                        norm_dist = dist_from_center / (np.sqrt(width**2 + height**2)/2)
                        spatial_region = 'core' if norm_dist < 0.3 else 'outskirts'
                
                if (metrics['mean'] >= min_brightness * 0.5 and  # Reduced by 50%
                    metrics['contrast'] >= min_contrast * 0.5 and  # Reduced by 50%
                    metrics['structure_score'] >= min_structure * 0.5 and  # Reduced by 50%
                    metrics['bg_ratio'] <= 0.98):  # Even more permissive
                    
                    # Calculate feature size factor properly placed here
                    feature_size = np.sum(tile_astro_mask) if tile_astro_mask is not None else 0
                    feature_size_factor = min(1.0, feature_size / (tile_width * tile_height * 0.05))  # Reduced from 0.08 to 0.05

                    # Add to quality score calculation with feature size factor
                    quality_score = (
                        metrics['structure_score'] * 0.35 +
                        metrics['contrast'] * 0.20 +
                        metrics['astro_coverage'] * 0.25 +
                        feature_size_factor * 0.15 +
                        metrics.get('dark_ratio', 0.0) * 0.05
                    )
                    
                    # Determine contrast bin
                    contrast_band = None
                    for band, (low, high) in contrast_bins.items():
                        if low <= metrics['contrast'] < high:
                            contrast_band = band
                            break
                    
                    # NEW: Determine brightness band
                    brightness_band = None
                    for band, (low, high) in brightness_bins.items():
                        if low <= metrics['mean'] < high:
                            brightness_band = band
                            break
                    
                    # Return the tile data
                    return {
                        'position': (x, y),
                        'size': (tile_width, tile_height),
                        'metrics': metrics,
                        'quality_score': quality_score,
                        'contrast_band': contrast_band,
                        'brightness_band': brightness_band,
                        'feature_type': feature_type,
                        'spatial_region': spatial_region,
                        'is_feature_centered': True  # Mark as feature-centered
                    }
            except Exception as e:
                # Skip tiles that cannot be extracted properly
                pass
            
            return None

        # Optimize batch processing
        print(f"Processing {len(sampled_centers)} features in parallel...")
        batch_size = 20  # Process in larger batches
        max_workers = min(16, os.cpu_count() or 4)  # More workers for better parallelism
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit in batches to reduce overhead
            all_feature_results = []
            for i in range(0, len(sampled_centers), batch_size):
                batch = sampled_centers[i:i+batch_size]
                futures = [executor.submit(process_feature_center, center) for center in batch]
                
                # Process results as they complete
                for future in as_completed(futures):
                    result = future.result()
                    if result is not None:
                        all_feature_results.append(result)
                
                # Print progress every batch
                print(f"Processed {min(i+batch_size, len(sampled_centers))}/{len(sampled_centers)}", flush=True)
            
            # Add results to all_tiles
            all_tiles.extend([r for r in all_feature_results if r is not None])

        print(f"Added {len(all_feature_results)} feature-centered tiles")

    # Then add regular grid-based tiles for complete coverage - OPTIMIZED
    # Use more aggressive adaptive sampling based on image size
    sampling_factor = 3  # Start with higher base sampling
    if max(height, width) > 2000:
        sampling_factor = 6
    if max(height, width) > 3000:
        sampling_factor = 9
    if max(height, width) > 4000:
        sampling_factor = 12
        
    print(f"Adding grid-based tiles with sampling factor {sampling_factor}...")
    
    # Skip grid-based tile estimation to save time
    # Go directly to quick sampling of grid positions
    positions_evaluated = 0
    grid_tiles_limit = 400  # Limit the number of grid-based tiles we'll evaluate
    
    # Batch the grid position evaluation for better performance
    grid_positions = []
    
    for tile_width, tile_height in tile_sizes:
        # Use adaptive step size
        step_x = max(1, (tile_width - overlap) * sampling_factor)
        step_y = max(1, (tile_height - overlap) * sampling_factor)
        
        # Collect all positions first
        for y in range(0, height - tile_height + 1, step_y):
            for x in range(0, width - tile_width + 1, step_x):
                if positions_evaluated >= grid_tiles_limit:
                    break
                
                grid_positions.append((x, y, tile_width, tile_height))
                positions_evaluated += 1
            
            if positions_evaluated >= grid_tiles_limit:
                break
    
    print(f"Evaluating {len(grid_positions)} grid positions in parallel...")
    
    def process_grid_position(position_data):
        x, y, tile_width, tile_height = position_data
        
        # Extract tile from original image
        try:
            orig_tile = image[y:y+tile_height, x:x+tile_width].copy()
            
            # Quick check for empty or flat tiles
            if len(orig_tile.shape) > 2 and orig_tile.shape[2] > 1:
                gray = np.mean(orig_tile, axis=2)
            else:
                gray = np.squeeze(orig_tile)
                
            # More permissive contrast check
            if np.max(gray) - np.min(gray) < min_contrast * 0.5:  # Reduced threshold by 50%
                return None
            
            # Get astro mask for this tile if available
            tile_astro_mask = None
            if astro_mask is not None:
                tile_astro_mask = astro_mask[y:y+tile_height, x:x+tile_width].copy()
            
            # Calculate metrics for quality assessment
            metrics = calculate_tile_metrics(orig_tile, tile_astro_mask)
            
            # Skip extremely dark areas that should be background
            if metrics['mean'] < 0.035 and metrics['edge_mean'] < 0.03:  # Reject very dark areas with minimal edges
                return None
            
            # NEW: Classify the tile feature type
            feature_type = classify_tile_type(metrics)
            
            # NEW: Determine spatial region
            spatial_region = 'outskirts'
            if astro_mask is not None:
                # Find astronomical object center approximation
                if np.any(astro_mask):
                    center_y_astro, center_x_astro = ndimage.center_of_mass(astro_mask)
                    # Calculate normalized distance from center
                    tile_center_y = y + tile_height/2
                    tile_center_x = x + tile_width/2
                    dist_from_center = np.sqrt((tile_center_x - center_x_astro)**2 + (tile_center_y - center_y_astro)**2)
                    norm_dist = dist_from_center / (np.sqrt(width**2 + height**2)/2)
                    spatial_region = 'core' if norm_dist < 0.3 else 'outskirts'
            
            # Use less strict quality thresholds but with a minimum quality floor
            if (metrics['mean'] >= min_brightness * 0.7 and 
                metrics['contrast'] >= min_contrast * 0.7 and 
                metrics['structure_score'] >= min_structure * 0.7 and 
                metrics['contrast'] >= 0.25 and  # Add a hard minimum for contrast
                metrics['bg_ratio'] <= 0.95):  # Increased from 0.92
                
                # Calculate quality score with emphasis on texture if available
                quality_score = (
                    metrics['structure_score'] * 0.35 +
                    metrics['contrast'] * 0.2 +
                    metrics['edge_mean'] * 0.15 +
                    metrics.get('texture_score', 0.0) * 0.1 +
                    metrics.get('dark_ratio', 0.0) * 0.1
                )
                
                # Add astronomical object quality if available
                if metrics['astro_coverage'] > 0:
                    quality_score += metrics['astro_coverage'] * 0.1
                
                # Determine contrast bin
                contrast_band = None
                for band, (low, high) in contrast_bins.items():
                    if low <= metrics['contrast'] < high:
                        contrast_band = band
                        break
                
                # NEW: Determine brightness band
                brightness_band = None
                for band, (low, high) in brightness_bins.items():
                    if low <= metrics['mean'] < high:
                        brightness_band = band
                        break
                
                # Return tile info
                return {
                    'position': (x, y),
                    'size': (tile_width, tile_height),
                    'metrics': metrics,
                    'quality_score': quality_score,
                    'contrast_band': contrast_band,
                    'brightness_band': brightness_band,
                    'feature_type': feature_type,
                    'spatial_region': spatial_region,
                    'is_feature_centered': False  # Regular grid tile
                }
        except Exception as e:
            # Skip tiles that cannot be extracted properly
            pass
        
        return None
    
    # Process grid positions in parallel
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit in batches to reduce overhead
        grid_results = []
        for i in range(0, len(grid_positions), batch_size):
            batch = grid_positions[i:i+batch_size]
            futures = [executor.submit(process_grid_position, pos) for pos in batch]
            
            # Process results as they complete
            for future in as_completed(futures):
                result = future.result()
                if result is not None:
                    grid_results.append(result)
            
            # Print progress occasionally
            if (i // batch_size) % 2 == 0:
                print(f"Processed {min(i+batch_size, len(grid_positions))}/{len(grid_positions)} grid positions", flush=True)
        
        # Add results to all_tiles
        all_tiles.extend([r for r in grid_results if r is not None])
    
    # Sort tiles by quality score
    print(f"Sorting {len(all_tiles)} candidate tiles...")
    all_tiles.sort(key=lambda x: x['quality_score'], reverse=True)
    
    # Select final tiles with diversity and minimal overlap
    print("Selecting final tiles...")
    selected_tiles = []
    selected_positions = set()

    # Count tiles per contrast bin, brightness bin, feature type, and region
    contrast_counts = {band: 0 for band in contrast_bins.keys()}
    brightness_counts = {band: 0 for band in brightness_bins.keys()}
    feature_type_counts = {ftype: 0 for ftype in feature_types.keys()}
    region_counts = {region: 0 for region in spatial_regions.keys()}

    # Calculate desired distribution
    desired_per_contrast = max_tiles // len(contrast_bins) if max_tiles else float('inf')
    desired_per_brightness = max_tiles // len(brightness_bins) if max_tiles else float('inf')
    desired_per_feature_type = max_tiles // len(feature_types) if max_tiles else float('inf')
    desired_per_region = max_tiles // 2 if max_tiles else float('inf')  # 2 for core/outskirts

    # First select feature-centered tiles
    feature_centered_tiles = [t for t in all_tiles if t.get('is_feature_centered', False)]
    print(f"Selecting from {len(feature_centered_tiles)} feature-centered tiles...")

    # Group tiles by feature type for balanced selection
    feature_type_groups = {}
    for tile in feature_centered_tiles:
        feature_type = tile.get('feature_type', 'undefined')
        if feature_type not in feature_type_groups:
            feature_type_groups[feature_type] = []
        feature_type_groups[feature_type].append(tile)

    print(f"Feature type distribution: {', '.join([f'{ft}: {len(tiles)}' for ft, tiles in feature_type_groups.items()])}")

    # Add this check right after printing the feature type distribution
    if len(feature_type_groups) == 0:
        print("No features found in this image. Returning empty tile lists.")
        return [], []  # Return empty lists if no features found

    # Select tiles with balanced feature types
    for feature_type, tiles in feature_type_groups.items():
        
        # Take at most the desired number per feature type
        for tile in tiles:
            if feature_type_counts.get(feature_type, 0) >= desired_per_feature_type:
                continue
                
            x, y = tile['position']
            w, h = tile['size']
            
            # Check for significant overlap with already selected tiles
            overlaps = False
            for sel_x, sel_y, sel_w, sel_h in selected_positions:
                # Calculate intersection
                x_overlap = max(0, min(x + w, sel_x + sel_w) - max(x, sel_x))
                y_overlap = max(0, min(y + h, sel_y + sel_h) - max(y, sel_y))
                overlap_area = x_overlap * y_overlap
                smaller_area = min(w * h, sel_w * sel_h)
                
                # Reduced overlap threshold to allow more tiles
                if overlap_area > 0.3 * smaller_area:  # Changed from 0.01 (extremely permissive) to more reasonable 0.3
                    overlaps = True
                    break
            
            if not overlaps:
                selected_tiles.append(tile)
                selected_positions.add((x, y, w, h))
                
                # Update distribution counts
                if 'contrast_band' in tile:
                    contrast_counts[tile['contrast_band']] = contrast_counts.get(tile['contrast_band'], 0) + 1
                if 'brightness_band' in tile:
                    brightness_counts[tile['brightness_band']] = brightness_counts.get(tile['brightness_band'], 0) + 1
                if 'feature_type' in tile:
                    feature_type_counts[tile['feature_type']] = feature_type_counts.get(tile['feature_type'], 0) + 1
                if 'spatial_region' in tile:
                    region_counts[tile['spatial_region']] = region_counts.get(tile['spatial_region'], 0) + 1
                
                # Break if we've reached max tiles
                if max_tiles and len(selected_tiles) >= max_tiles:
                    break
        
        # Break if we've reached max tiles
        if max_tiles and len(selected_tiles) >= max_tiles:
            break

    # Then select additional tiles to balance contrast, brightness, and region
    print(f"Selected {len(selected_tiles)} feature-centered tiles")
    print(f"Current distribution - Contrast: {contrast_counts}, Brightness: {brightness_counts}, Features: {feature_type_counts}, Regions: {region_counts}")

    # Calculate which categories need more samples
    need_more = {}
    for feature_type, count in feature_type_counts.items():
        if count < desired_per_feature_type * 0.7:  # If we have less than 70% of target
            need_more[feature_type] = desired_per_feature_type - count

    for brightness_band, count in brightness_counts.items():
        if count < desired_per_brightness * 0.7:
            need_more[brightness_band] = desired_per_brightness - count

    for contrast_band, count in contrast_counts.items():
        if count < desired_per_contrast * 0.7:
            need_more[contrast_band] = desired_per_contrast - count

    for region, count in region_counts.items():
        if count < desired_per_region * 0.7:
            need_more[region] = desired_per_region - count

    print(f"Categories needing more samples: {need_more}")

    # Force at least some tiles from each feature type by lowering thresholds
    for feature_type in feature_types.keys():
        if feature_type_counts.get(feature_type, 0) == 0:
            # Find any tiles of this type, even if lower quality
            type_candidates = [t for t in all_tiles if t.get('feature_type', '') == feature_type]
            if type_candidates:
                # Take up to 5 of each missing type
                for i, tile in enumerate(type_candidates[:5]):
                    x, y = tile['position']
                    w, h = tile['size']
                    selected_tiles.append(tile)
                    selected_positions.add((x, y, w, h))
                    if 'feature_type' in tile:
                        feature_type_counts[tile['feature_type']] = feature_type_counts.get(tile['feature_type'], 0) + 1
                print(f"Added {min(5, len(type_candidates))} forced tiles of type {feature_type}")

    # Filter regular tiles to focus on underrepresented categories
    regular_tiles = [t for t in all_tiles if not t.get('is_feature_centered', False)]
    prioritized_tiles = []

    for tile in regular_tiles:
        priority_score = 0
        
        # Check each category this tile belongs to
        categories = [
            tile.get('feature_type', 'undefined'),
            tile.get('brightness_band', 'undefined'),
            tile.get('contrast_band', 'undefined'),
            tile.get('spatial_region', 'undefined')
        ]
        
        # Calculate priority based on need
        for category in categories:
            if category in need_more:
                priority_score += need_more[category]
        
        if priority_score > 0:
            tile['priority_score'] = priority_score
            prioritized_tiles.append(tile)

    # Sort by priority score
    prioritized_tiles.sort(key=lambda x: x.get('priority_score', 0), reverse=True)

    # Select prioritized tiles
    print(f"Selecting from {len(prioritized_tiles)} prioritized regular tiles...")
    for tile in prioritized_tiles:
        if max_tiles and len(selected_tiles) >= max_tiles:
            break
            
        x, y = tile['position']
        w, h = tile['size']
        
        # Check for overlap
        overlaps = False
        for sel_x, sel_y, sel_w, sel_h in selected_positions:
            x_overlap = max(0, min(x + w, sel_x + sel_w) - max(x, sel_x))
            y_overlap = max(0, min(y + h, sel_y + sel_h) - max(y, sel_y))
            overlap_area = x_overlap * y_overlap
            smaller_area = min(w * h, sel_w * sel_h)
            
            if overlap_area > 0.3 * smaller_area:
                overlaps = True
                break
        
        if not overlaps:
            selected_tiles.append(tile)
            selected_positions.add((x, y, w, h))
            
            # Update distribution counts
            if 'contrast_band' in tile:
                contrast_counts[tile['contrast_band']] = contrast_counts.get(tile['contrast_band'], 0) + 1
            if 'brightness_band' in tile:
                brightness_counts[tile['brightness_band']] = brightness_counts.get(tile['brightness_band'], 0) + 1
            if 'feature_type' in tile:
                feature_type_counts[tile['feature_type']] = feature_type_counts.get(tile['feature_type'], 0) + 1
            if 'spatial_region' in tile:
                region_counts[tile['spatial_region']] = region_counts.get(tile['spatial_region'], 0) + 1

    print(f"Selected {len(selected_tiles)} total tiles")
    print(f"Final distribution - Contrast: {contrast_counts}, Brightness: {brightness_counts}, Features: {feature_type_counts}, Regions: {region_counts}")
    
    # If we didn't get enough tiles, reduce requirements and try again
    if max_tiles and len(selected_tiles) < max_tiles * 0.8:  # If we got less than 80% of target
        print(f"Only found {len(selected_tiles)} tiles, adding more with diversity-focused selection...")
        
        # Use all tiles as potential candidates for maximum options
        remaining_tiles = all_tiles.copy()
        
        # Sort remaining tiles by quality score
        remaining_tiles.sort(key=lambda x: x['quality_score'], reverse=True)
        
        # Calculate existing tile centers for diversity measurement
        existing_centers = []
        for sel_x, sel_y, sel_w, sel_h in selected_positions:
            center_x = sel_x + sel_w // 2
            center_y = sel_y + sel_h // 2
            existing_centers.append((center_x, center_y))
        
        # NEW: Add a diversity score to each remaining tile
        for tile in remaining_tiles:
            x, y = tile['position']
            w, h = tile['size']
            center_x = x + w // 2
            center_y = y + h // 2
            
            # Calculate minimum distance to any existing tile
            min_distance = float('inf')
            for ex_x, ex_y in existing_centers:
                # Calculate Euclidean distance between centers
                distance = ((center_x - ex_x) ** 2 + (center_y - ex_y) ** 2) ** 0.5
                min_distance = min(min_distance, distance)
            
            # Normalize distance based on image size (diagonal)
            image_diagonal = (height ** 2 + width ** 2) ** 0.5
            normalized_distance = min_distance / image_diagonal
            
            # Combine quality score with diversity for overall ranking
            # Weight distance more heavily to encourage diverse selections
            tile['combined_score'] = tile['quality_score'] * 0.3 + normalized_distance * 0.7
        
        # Re-sort remaining tiles by combined score
        remaining_tiles.sort(key=lambda x: x.get('combined_score', 0), reverse=True)
        
        # Try adding more tiles with improved diversity
        for tile in remaining_tiles:
            if max_tiles and len(selected_tiles) >= max_tiles:
                break
                
            x, y = tile['position']
            w, h = tile['size']
            
            # Less permissive overlap check
            overlaps = False
            for sel_x, sel_y, sel_w, sel_h in selected_positions:
                # Calculate intersection
                x_overlap = max(0, min(x + w, sel_x + sel_w) - max(x, sel_x))
                y_overlap = max(0, min(y + h, sel_y + sel_h) - max(y, sel_y))
                overlap_area = x_overlap * y_overlap
                smaller_area = min(w * h, sel_w * sel_h)
                
                # Much stricter overlap threshold
                if overlap_area > 0.3 * smaller_area:  # Reduced from 0.8 to 0.3 (70% must be unique)
                    overlaps = True
                    break
            
            if not overlaps:
                # Extract the actual tile from the original image
                try:
                    # If this is a unique position, add it
                    if (x, y, w, h) not in selected_positions:
                        selected_tiles.append(tile)
                        selected_positions.add((x, y, w, h))
                        
                        # Update contrast band counts if relevant
                        if tile['contrast_band'] and tile['contrast_band'] in contrast_counts:
                            contrast_counts[tile['contrast_band']] += 1
                        
                        # NEW: Update existing centers for future diversity calculations
                        existing_centers.append((x + w // 2, y + h // 2))
                except Exception as e:
                    # Skip tiles with extraction issues
                    pass
        
        print(f"After diversity-focused selection: {len(selected_tiles)} tiles")
    
    # Convert to actual tile objects with image data
    accepted_tiles = []
    
    # NEW: Randomize the order of tiles slightly to increase diversity in selected tiles
    # This helps when max_tiles limits selection and tiles have similar quality scores
    if max_tiles and len(selected_tiles) > max_tiles * 1.2:  # If we have 20% more tiles than needed
        # Keep top 30% of tiles by quality
        top_tiles = sorted(selected_tiles, key=lambda x: x['quality_score'], reverse=True)[:int(max_tiles * 0.3)]
        # Randomly sample the rest to get variety
        remaining = [t for t in selected_tiles if t not in top_tiles]
        random_selection = random.sample(remaining, min(len(remaining), max_tiles - len(top_tiles)))
        selected_tiles = top_tiles + random_selection
    
    for tile_info in selected_tiles:
        x, y = tile_info['position']
        w, h = tile_info['size']
        
        # Extract the actual tile from the original image
        try:
            orig_tile = image[y:y+h, x:x+w].copy()
            
            # Create final tile info
            final_tile = {
                'tile': orig_tile,
                'metrics': tile_info['metrics'],
                'position': (x, y),
                'original_size': (w, h),
                'quality_score': tile_info['quality_score']
            }
            
            accepted_tiles.append(final_tile)
        except Exception as e:
            print(f"Error extracting tile at ({x}, {y}): {e}")
    
    # Return empty list for rejected tiles (not important for main processing)
    rejected_tiles = []
    
    return accepted_tiles, rejected_tiles

def create_blurred_versions(tile, normalize_back=True, orig_range=None, add_noise=True, 
                           noise_preservation=0.4, edge_aware=False, beta=4.0, 
                           denoise_ground_truth=True, gt_denoise_strength=0.8):
    """
    Correct camera physics: Blur happens optically, noise happens at sensor
    Both target and input get SAME noise, only spatial detail differs
    """
    orig_dtype = tile.dtype
    tile = tile.astype(np.float32)
    
    # Store original range
    if normalize_back:
        if orig_range is None:
            orig_min, orig_max = np.min(tile), np.max(tile)
            orig_range = (orig_min, orig_max)
        else:
            orig_min, orig_max = orig_range
    else:
        orig_min, orig_max = 0, 1
    
    original = tile.copy()
    
    # STEP 1: Create clean signal (denoised)
    if denoise_ground_truth:
        clean_signal = denoise_image(original, strength=gt_denoise_strength)
        clean_signal = denoise_image(clean_signal, strength=gt_denoise_strength * 0.4)
    else:
        clean_signal = original.copy()
    
    # STEP 2: Generate ONE noise pattern to add to ALL images
    base_noise_params = generate_noise_parameters(clean_signal)
    noise_pattern = generate_camera_noise_with_params(clean_signal, base_noise_params, noise_factor=1.0)
    
    # STEP 3: Define blur levels
    blur_levels = [
        max(1.8, 2.2 * (tile.shape[0] / 256)),   # Light blur
        max(3.0, 3.8 * (tile.shape[0] / 256)),   # Moderate blur  
        max(4.5, 5.5 * (tile.shape[0] / 256)),   # Strong blur
        max(6.0, 8.0 * (tile.shape[0] / 256))    # Very strong blur
    ]
    
    def create_moffat_kernel(fwhm, beta_value=beta):
        beta_variation = np.random.uniform(0.9, 1.1)
        beta_value = beta_value * beta_variation
        alpha = fwhm / (2 * np.sqrt(2**(1/beta_value) - 1))
        
        size = int(np.ceil(4 * fwhm))
        if size % 2 == 0:
            size += 1
        size = max(5, size)
        
        x = np.arange(0, size) - (size - 1) / 2
        y = x[:, np.newaxis]
        r = np.sqrt(x**2 + y**2)
        
        kernel = (1 + (r/alpha)**2)**(-beta_value)
        return kernel / kernel.sum()
    
    def moffat_blur_with_chromatic(image, fwhm, beta_value=beta):
        if len(image.shape) > 2 and image.shape[2] > 1:
            result = np.zeros_like(image)
            for c in range(image.shape[2]):
                chromatic_factor = 1.0 + 0.02 * (c - 1)
                channel_fwhm = fwhm * chromatic_factor
                kernel = create_moffat_kernel(channel_fwhm, beta_value)
                result[:,:,c] = cv2.filter2D(image[:,:,c], -1, kernel)
            return result
        else:
            kernel = create_moffat_kernel(fwhm, beta_value)
            return cv2.filter2D(image, -1, kernel)
    
    # STEP 4: Create blurred versions by blurring CLEAN signal, then adding noise
    blurred_versions = []
    for i, blur_fwhm in enumerate(blur_levels):
        fwhm_variation = np.random.uniform(0.95, 1.05)
        actual_fwhm = blur_fwhm * fwhm_variation
        
        # 1. Blur the CLEAN signal (optical blur happens first)
        blurred_clean = moffat_blur_with_chromatic(clean_signal, actual_fwhm)
        
        # 2. Add the SAME noise pattern (sensor noise happens after optics)
        blurred_final = np.clip(blurred_clean + noise_pattern, 0, 1)
        
        # Apply range normalization
        if normalize_back and orig_range is not None:
            blurred_final = blurred_final * (orig_max - orig_min) + orig_min
            if np.issubdtype(orig_dtype, np.integer):
                blurred_final = np.clip(blurred_final, orig_min - 0.5, orig_max + 0.5)
        
        blurred_versions.append(blurred_final.astype(orig_dtype))
    
    # STEP 5: Ground truth = clean signal + SAME noise pattern
    ground_truth = np.clip(clean_signal + noise_pattern, 0, 1)
    
    # Apply range normalization to ground truth
    if normalize_back and orig_range is not None:
        ground_truth_final = ground_truth * (orig_max - orig_min) + orig_min
        if np.issubdtype(orig_dtype, np.integer):
            ground_truth_final = np.clip(ground_truth_final, orig_min - 0.5, orig_max + 0.5)
    else:
        ground_truth_final = ground_truth
    
    return blurred_versions, ground_truth_final.astype(orig_dtype)


def generate_camera_noise_with_params(image, params, noise_factor=1.0):
    """
    Generate slightly more visible noise - final tuning
    """
    np.random.seed(params['hot_pixel_seed'])
    
    noise_total = np.zeros_like(image)
    
    # Just a tad more noise for perfect visibility
    base_photon_scale = 7000   # Reduced from 8000 (bit more noise)
    base_read_noise = 0.022    # Increased from 0.018 (bit more noise)
    base_color_noise = 0.014   # Increased from 0.012 (bit more noise)
    
    # 1. Photon (Shot) Noise
    photon_scale = base_photon_scale / noise_factor
    for c in range(image.shape[2] if len(image.shape) > 2 else 1):
        if len(image.shape) > 2:
            signal = image[:,:,c]
        else:
            signal = image
            
        signal_photons = np.maximum(signal * photon_scale, 1)
        photon_noise = (np.random.poisson(signal_photons) - signal_photons) / photon_scale
        
        if len(image.shape) > 2:
            noise_total[:,:,c] += photon_noise * 0.85 * noise_factor  # Increased from 0.8
        else:
            noise_total += photon_noise * 0.85 * noise_factor
    
    # 2. Read Noise
    read_noise_std = base_read_noise * noise_factor
    read_noise = np.random.normal(0, read_noise_std, image.shape)
    noise_total += read_noise
    
    # 3. Color-specific noise
    if len(image.shape) > 2 and image.shape[2] >= 3:
        for c in range(image.shape[2]):
            channel_noise_factor = 1.0 + (c - 1) * 0.10  # Increased from 0.08
            channel_noise = np.random.normal(0, base_color_noise * noise_factor * channel_noise_factor, image.shape[:2])
            noise_total[:,:,c] += channel_noise
    
    # 4. Dark Current Noise
    dark_current_level = 0.006 * noise_factor  # Increased from 0.005
    dark_noise = np.random.poisson(dark_current_level * 1000, image.shape) / 1000.0 - dark_current_level
    noise_total += dark_noise * 0.5
    
    # 5. Fixed Pattern Noise
    np.random.seed(params['pattern_seed'])
    if np.random.random() > 0.5:
        pattern_amplitude = 0.007 * noise_factor  # Increased from 0.006
        h, w = image.shape[:2]
        pattern_freq = np.random.uniform(0.1, 0.3)
        x = np.linspace(0, pattern_freq * 2 * np.pi, w)
        y = np.linspace(0, pattern_freq * 2 * np.pi, h)
        X, Y = np.meshgrid(x, y)
        pattern = pattern_amplitude * np.sin(X) * np.cos(Y)
        
        if len(image.shape) > 2:
            for c in range(image.shape[2]):
                noise_total[:,:,c] += pattern
        else:
            noise_total += pattern
    
    # 6. Hot pixels
    if len(image.shape) > 2:
        for c in range(image.shape[2]):
            num_hot = max(1, int(0.00025 * image.shape[0] * image.shape[1]))  # Increased from 0.0002
            hot_y = np.random.randint(0, image.shape[0], num_hot)
            hot_x = np.random.randint(0, image.shape[1], num_hot)
            hot_val = np.random.uniform(0.12, 0.28, num_hot) * noise_factor  # Increased from 0.10, 0.25
            noise_total[hot_y, hot_x, c] += hot_val
    
    # 7. High-frequency noise
    high_freq_noise = np.random.normal(0, 0.010 * noise_factor, image.shape)  # Increased from 0.008
    noise_total += high_freq_noise
    
    np.random.seed()
    return noise_total

def generate_noise_parameters(image):
    """Updated noise parameters"""
    return {
        'photon_scale': 7000,      # Reduced for more noise
        'read_noise_std': 0.022,   # Increased for more noise
        'dark_current_level': 0.006,
        'pattern_seed': np.random.randint(0, 10000),
        'hot_pixel_seed': np.random.randint(0, 10000)
    }

def apply_augmentations(tile, num_augmentations=3):
    """
    Apply augmentations including grayscale and color variations
    without requiring additional data
    """
    # Start with the original tile
    augmented_tiles = [tile.copy()]
    
    # We'll use a balanced approach for the augmentations
    for i in range(num_augmentations):
        # Choose random augmentations
        flip_horizontal = random.choice([True, False])
        flip_vertical = random.choice([True, False])
        rotation_angle = random.choice([0, 90, 180, 270])
        
        # Apply augmentations
        augmented = tile.copy()
        
        # Apply spatial transformations
        if rotation_angle > 0:
            augmented = np.rot90(augmented, k=rotation_angle // 90)
        if flip_horizontal:
            augmented = np.fliplr(augmented)
        if flip_vertical:
            augmented = np.flipud(augmented)
        
        # For every third augmentation (33%), convert to grayscale
        # This gives us a good balance without needing more data
        if i % 3 == 0 and augmented.shape[2] == 3:
            # Convert to grayscale but keep 3 channels for consistency
            gray = np.mean(augmented, axis=2, keepdims=True)
            augmented = np.repeat(gray, 3, axis=2)
            
        augmented_tiles.append(augmented)
    
    return augmented_tiles

def create_diverse_training_pairs(tile, num_variations=3):
    """
    Create diverse blur variations beyond simple Moffat blur
    Simulates various optical and atmospheric conditions
    """
    variations = []
    
    # Get tile size for scaling
    size = tile.shape[0]
    scale_factor = size / 256
    
    for i in range(num_variations):
        # Randomly select a degradation type
        degradation_type = np.random.choice(['moffat', 'gaussian', 'motion', 'defocus', 'atmospheric'])
        
        if degradation_type == 'moffat':
            # Standard Moffat with variation
            beta = np.random.uniform(2.0, 4.5)
            fwhm = np.random.uniform(1.5, 4.0) * scale_factor
            blurred = create_moffat_blur(tile, fwhm, beta)
            
        elif degradation_type == 'gaussian':
            # Atmospheric seeing
            sigma = np.random.uniform(1.0, 3.0) * scale_factor
            blurred = cv2.GaussianBlur(tile, (0, 0), sigmaX=sigma)
            
        elif degradation_type == 'motion':
            # Tracking errors
            angle = np.random.uniform(0, 360)
            length = np.random.uniform(3, 10) * scale_factor
            blurred = apply_motion_blur(tile, angle, length)
            
        elif degradation_type == 'defocus':
            # Focus errors
            radius = np.random.uniform(2, 5) * scale_factor
            blurred = apply_defocus_blur(tile, radius)
            
        elif degradation_type == 'atmospheric':
            # Complex atmospheric effects
            # Combination of multiple effects
            sigma1 = np.random.uniform(0.5, 1.5) * scale_factor
            sigma2 = np.random.uniform(2.0, 3.0) * scale_factor
            weight = np.random.uniform(0.3, 0.7)
            
            blur1 = cv2.GaussianBlur(tile, (0, 0), sigmaX=sigma1)
            blur2 = cv2.GaussianBlur(tile, (0, 0), sigmaX=sigma2)
            blurred = weight * blur1 + (1-weight) * blur2
        
        # Add realistic noise
        blurred = add_photon_noise(blurred, preserve_snr=True)
        
        variations.append({
            'image': blurred,
            'type': degradation_type,
            'parameters': {'variation_idx': i}
        })
    
    return variations

def apply_motion_blur(image, angle, length):
    """Apply motion blur at specified angle"""
    # Create motion blur kernel
    kernel = np.zeros((int(length), int(length)))
    center = length // 2
    
    # Draw line at angle
    angle_rad = np.deg2rad(angle)
    for i in range(int(length)):
        offset = i - center
        x = int(center + offset * np.cos(angle_rad))
        y = int(center + offset * np.sin(angle_rad))
        if 0 <= x < length and 0 <= y < length:
            kernel[y, x] = 1
    
    # Normalize
    kernel = kernel / np.sum(kernel)
    
    # Apply to each channel
    if len(image.shape) > 2:
        result = np.zeros_like(image)
        for c in range(image.shape[2]):
            result[:,:,c] = cv2.filter2D(image[:,:,c], -1, kernel)
        return result
    else:
        return cv2.filter2D(image, -1, kernel)

def apply_defocus_blur(image, radius):
    """Apply circular defocus blur"""
    # Create circular kernel
    kernel_size = int(2 * radius + 1)
    kernel = np.zeros((kernel_size, kernel_size))
    center = kernel_size // 2
    
    # Create circle
    y, x = np.ogrid[:kernel_size, :kernel_size]
    mask = (x - center)**2 + (y - center)**2 <= radius**2
    kernel[mask] = 1
    kernel = kernel / np.sum(kernel)
    
    # Apply to each channel
    if len(image.shape) > 2:
        result = np.zeros_like(image)
        for c in range(image.shape[2]):
            result[:,:,c] = cv2.filter2D(image[:,:,c], -1, kernel)
        return result
    else:
        return cv2.filter2D(image, -1, kernel)

def create_moffat_blur(image, fwhm, beta):
    """Helper function for Moffat blur (extracted from create_blurred_versions)"""
    # Calculate alpha from FWHM and beta
    alpha = fwhm / (2 * np.sqrt(2**(1/beta) - 1))
    
    # Calculate kernel size
    size = int(np.ceil(4 * fwhm))
    if size % 2 == 0:
        size += 1
    size = max(5, size)
    
    # Create coordinate grid
    x = np.arange(0, size) - (size - 1) / 2
    y = x[:, np.newaxis]
    r = np.sqrt(x**2 + y**2)
    
    # Calculate Moffat profile
    kernel = (1 + (r/alpha)**2)**(-beta)
    kernel = kernel / kernel.sum()
    
    # Apply to image
    if len(image.shape) > 2:
        result = np.zeros_like(image)
        for c in range(image.shape[2]):
            result[:,:,c] = cv2.filter2D(image[:,:,c], -1, kernel)
        return result
    else:
        return cv2.filter2D(image, -1, kernel)   

def save_tile_set(ground_truth, blurred_versions, base_name, output_dirs, orig_range=None, is_clean_pair=False):
    """
    Save a set of tiles (ground truth and blurred versions) to the output directories
    Now handles clean ground truth properly
    """
    # Create subdirectories if they don't exist
    for dir_name in output_dirs.values():
        os.makedirs(dir_name, exist_ok=True)
    
    # Save ground truth (now properly denoised)
    gt_path = os.path.join(output_dirs['ground_truth'], f"{base_name}.tif")
    tifffile.imwrite(gt_path, ground_truth.astype(np.float32))
    
    # Save blurred versions
    blur_paths = []
    for i, blurred in enumerate(blurred_versions):
        blur_key = f'blur_{i+1}'
        if blur_key in output_dirs:
            blur_path = os.path.join(output_dirs[blur_key], f"{base_name}.tif")
            tifffile.imwrite(blur_path, blurred.astype(np.float32))
            blur_paths.append(blur_path)
    
    # For clean-clean pairs, also save the ground truth as a "blur" version
    # This teaches the model that sometimes no correction is needed
    if is_clean_pair:
        # Save ground truth as blur_1 to create input=output pair
        clean_blur_path = os.path.join(output_dirs['blur_1'], f"{base_name}_clean.tif")
        tifffile.imwrite(clean_blur_path, ground_truth.astype(np.float32))
        blur_paths.append(clean_blur_path)
    
    # Return paths for verification
    return {
        'ground_truth': gt_path,
        'blurred': blur_paths,
        'is_clean_pair': is_clean_pair
    }

def visualize_tile_samples(accepted_tiles, rejected_tiles, output_dir):
    """
    Create debug visualizations for accepted and rejected tiles, now showing all 4 blur levels
    """
    debug_dir = os.path.join(output_dir, 'Debug')
    os.makedirs(debug_dir, exist_ok=True)
    
    # Sample a few tiles for visualization
    num_samples = min(10, len(accepted_tiles))
    accepted_samples = random.sample(accepted_tiles, num_samples) if num_samples > 0 else []
    
    # Create visualization of accepted tiles
    if accepted_samples:
        plt.figure(figsize=(15, 5))
        plt.suptitle("Sample of Accepted Tiles")
        
        for i, tile_info in enumerate(accepted_samples[:5]):
            plt.subplot(1, 5, i+1)
            # Display a resize of the tile for visualization
            tile = tile_info['tile']
            tile_display = cv2.resize(tile, (256, 256), interpolation=cv2.INTER_CUBIC)
            plt.imshow(np.clip(tile_display, 0, 1))
            
            # Include quality score and contrast in title
            contrast = tile_info['metrics']['contrast']
            quality = tile_info.get('quality_score', 0)
            plt.title(f"Size: {tile_info['original_size'][0]}px\nQuality: {quality:.2f}\nContrast: {contrast:.2f}")
            plt.axis('off')
            
        plt.tight_layout()
        plt.savefig(os.path.join(debug_dir, "accepted_tiles.png"))
        plt.close()

    # Create visualization of blur levels
    if accepted_samples:
        # Select a sample tile and resize to 256x256 for visualization
        sample_tile_orig = accepted_samples[0]['tile']
        sample_tile = cv2.resize(sample_tile_orig, (256, 256), interpolation=cv2.INTER_CUBIC)
        
        # Apply light denoising to ground truth
        denoised_tile = denoise_image(sample_tile, strength=0.15)  # Very light denoising
        
        # Generate blurred versions from the lightly denoised sample
        blurred_versions, clean_gt = create_blurred_versions(
            denoised_tile,
            denoise_ground_truth=True,
            gt_denoise_strength=0.15
        )
        
        # Create larger figure to show all 6 versions (original, denoised, and 4 blur levels)
        plt.figure(figsize=(18, 6))
        plt.suptitle("Blur Progression Example")
        
        plt.subplot(1, 6, 1)
        plt.imshow(np.clip(sample_tile, 0, 1))
        plt.title("Original")
        plt.axis('off')
        
        plt.subplot(1, 6, 2)
        plt.imshow(np.clip(clean_gt, 0, 1))
        plt.title("Ground Truth\n(Denoised)")
        plt.axis('off')
        
        for i, blurred in enumerate(blurred_versions):
            plt.subplot(1, 6, i+3)
            plt.imshow(np.clip(blurred, 0, 1))
            plt.title(f"Blur Level {i+1}")
            plt.axis('off')
            
        plt.tight_layout()
        plt.savefig(os.path.join(debug_dir, "blur_levels.png"))
        plt.close()

def generate_dataset(args):
    """Main function to generate the dataset"""
    # Create output directory structure if not in estimate-only mode
    if not args.estimate_only:
        output_dirs = create_directory_structure(args.output_dir)
    else:
        output_dirs = None
    
    # Collect all TIFF files in the input directory
    image_paths = []
    for ext in ['.tif', '.tiff']:
        image_paths.extend(glob.glob(os.path.join(args.input_dir, f"*{ext}")))
    
    print(f"Found {len(image_paths)} TIFF images in {args.input_dir}")
    
    # Still track validation split for reporting, but use same directories
    random.seed(42)  # For reproducibility
    validation_images = set(random.sample(image_paths, max(1, int(len(image_paths) * 0.1))))
    print(f"Created validation split with {len(validation_images)} images (using same directories)")
    
    # Dictionary to store mapping between tiles
    tile_mapping = {}
    total_tiles_generated = 0
    total_accepted_tiles = 0  # Track accepted tiles for estimation
    total_background_patches = 0  # Track background patches
    
    # Create debug directory if needed
    debug_dir = os.path.join(args.output_dir, 'Debug')
    if args.debug:
        os.makedirs(debug_dir, exist_ok=True)
    
    # Process each image
    for img_idx, image_path in enumerate(tqdm(image_paths, desc="Processing images", mininterval=0.1)):
        image_name = os.path.splitext(os.path.basename(image_path))[0]
        
        # Determine if this image is for training or validation (for reporting only)
        if image_path in validation_images:
            print(f"\nProcessing VALIDATION image {img_idx+1}/{len(image_paths)}: {image_path}")
        else:
            print(f"\nProcessing TRAINING image {img_idx+1}/{len(image_paths)}: {image_path}")
        
        # Load the image
        image, orig_range = load_tiff_image(image_path)
        if image is None:
            print(f"Skipping {image_path} - failed to load")
            continue
            
        print(f"Image shape: {image.shape}, Range: {orig_range}")

        # Detect astronomical features if requested - OPTIMIZED VERSION
        astro_mask = None
        if args.prefer_astro_features:
            # Always aggressively downsample for feature detection
            h, w = image.shape[:2]
            # Use consistent downsampling to 1000px max dim for faster feature detection
            max_dim = 1000
            scale_factor = max_dim / max(h, w)
            small_h = max(int(h * scale_factor), 100)
            small_w = max(int(w * scale_factor), 100)
            small_image = cv2.resize(image, (small_w, small_h), interpolation=cv2.INTER_AREA)
            
            # Detect on smaller image
            small_mask = detect_astronomical_features(small_image)
            
            # Resize mask back to original size
            astro_mask = cv2.resize(small_mask.astype(np.uint8), 
                                   (w, h), 
                                   interpolation=cv2.INTER_NEAREST).astype(bool)
            print(f"Used downsampled image ({small_w}x{small_h}) for feature detection")
            
            # Save a debug visualization if requested
            if args.debug:
                plt.figure(figsize=(10, 5))
                plt.subplot(1, 2, 1)
                plt.imshow(np.clip(image, 0, 1))
                plt.title("Original Image")
                plt.axis('off')
                
                plt.subplot(1, 2, 2)
                plt.imshow(np.clip(image, 0, 1))
                plt.imshow(astro_mask, alpha=0.3, cmap='Reds')
                plt.title("Astronomical Features")
                plt.axis('off')
                
                plt.tight_layout()
                plt.savefig(os.path.join(debug_dir, f"{image_name}_features.png"))
                plt.close()

        # Extract and save background samples
        if not args.estimate_only and not args.tiles_only:
            # Extract background samples
            print("Extracting background samples...")
            background_patches = extract_background_samples(
                image, 
                astro_mask, 
                num_samples=20,
                patch_size=args.tile_size
            )
            
            # Save background patches
            print(f"Extracted {len(background_patches)} background samples")
            total_background_patches += len(background_patches)
            
            if len(background_patches) > 0 and output_dirs is not None:
                for bg_idx, bg_patch in enumerate(background_patches):
                    bg_id = f"{image_name}_bg_{bg_idx}_{uuid.uuid4().hex[:8]}"
                    bg_path = os.path.join(output_dirs['background'], f"{bg_id}.tif")
                    tifffile.imwrite(bg_path, bg_patch['patch'].astype(np.float32))
                    
                    # Save debug visualization if requested
                    if args.debug and bg_idx < 5:  # Just show first 5
                        plt.figure(figsize=(5, 5))
                        plt.imshow(np.clip(bg_patch['patch'], 0, 1))
                        plt.title(f"Background Sample {bg_idx+1}")
                        plt.axis('off')
                        plt.savefig(os.path.join(debug_dir, f"{image_name}_bg_{bg_idx}.png"))
                        plt.close()
        
        accepted_tiles, rejected_tiles = select_tiles_intelligently(
            image,
            astro_mask,
            tile_size=args.tile_size, 
            overlap=args.overlap,
            min_brightness=args.min_brightness * 0.7,  # Make 30% more permissive
            min_structure=args.min_structure * 0.7,    # Make 30% more permissive
            max_tiles=args.max_tiles,
            min_contrast=args.min_contrast * 0.7,      # Make 30% more permissive
            min_extraction_size=args.min_extraction_size,
            max_extraction_size=args.max_extraction_size
        )
        
        # Update total accepted tiles count
        total_accepted_tiles += len(accepted_tiles)
        
        # Save a debug visualization of sample tiles
        if args.debug and len(accepted_tiles) > 0:
            visualize_tile_samples(accepted_tiles, rejected_tiles, args.output_dir)
            
            # Create a grid of sample tiles (5x5 or fewer)
            sample_size = min(25, len(accepted_tiles))
            if sample_size > 0:
                samples = random.sample(accepted_tiles, sample_size)
                
                grid_size = int(np.ceil(np.sqrt(sample_size)))
                grid = np.zeros((grid_size * args.tile_size, grid_size * args.tile_size, 3), dtype=np.float32)
                
                for i, tile_info in enumerate(samples):
                    row = i // grid_size
                    col = i % grid_size
                    
                    # Resize tile to the target size
                    tile_resized = cv2.resize(tile_info['tile'], (args.tile_size, args.tile_size), 
                                            interpolation=cv2.INTER_CUBIC)
                    
                    tile_img = np.clip(tile_resized, 0, 1)
                    grid[row*args.tile_size:(row+1)*args.tile_size, 
                         col*args.tile_size:(col+1)*args.tile_size] = tile_img
                
                plt.figure(figsize=(12, 12))
                plt.imshow(grid)
                plt.title(f"Sample of {sample_size} Accepted Tiles from {image_name}")
                plt.axis('off')
                plt.savefig(os.path.join(debug_dir, f"{image_name}_sample_tiles.png"))
                plt.close()
        
        # Skip the rest if we're only estimating or just selecting tiles
        if args.estimate_only or args.tiles_only:
            # For tiles-only mode, save the tile information to a JSON file
            if args.tiles_only and len(accepted_tiles) > 0:
                tile_info_file = os.path.join(args.output_dir, f"{image_name}_tiles.json")
                tile_infos = []
                for idx, tile in enumerate(accepted_tiles):
                    tile_infos.append({
                        'image_name': image_name,
                        'position': tile['position'],
                        'size': tile['original_size'],
                        'quality_score': float(tile['quality_score']),
                        'metrics': {k: float(v) if isinstance(v, (int, float)) else v 
                                for k, v in tile['metrics'].items()}
                    })
                with open(tile_info_file, 'w') as f:
                    json.dump(tile_infos, f, indent=2)
                print(f"Saved {len(accepted_tiles)} tile positions to {tile_info_file}")
            continue
        
        # Fast parallel processing of accepted tiles when there are many
        from concurrent.futures import ThreadPoolExecutor
        from functools import partial
        
        def process_tile(tile_info, tile_idx, augmentations, tile_size, output_dirs, orig_range, 
                        image_name, denoise_strength=0.2, noise_preservation=0.6, edge_aware=False, beta=4.0):
            """Process a tile with proper ground truth denoising and clean-clean pairs"""
            # Get the tile
            orig_tile = tile_info['tile']
            
            # Ensure square aspect ratio with proper handling
            h, w = orig_tile.shape[:2]
            
            # Check for non-square tiles
            if w != h:
                # Create square tile to ensure proper aspect ratio
                max_dim = max(w, h)
                square_tile = np.zeros((max_dim, max_dim, orig_tile.shape[2]), dtype=orig_tile.dtype)
                
                # Center the tile in the square
                start_x = (max_dim - w) // 2
                start_y = (max_dim - h) // 2
                square_tile[start_y:start_y+h, start_x:start_x+w] = orig_tile
                
                # Use square tile for processing
                orig_tile = square_tile
            
            # Now resize to the target size
            tile = cv2.resize(orig_tile, (tile_size, tile_size), interpolation=cv2.INTER_CUBIC)
            
            # Apply augmentations
            augmented_tiles = apply_augmentations(tile, augmentations)
            
            result_tiles = []
            # Process each augmented tile
            for aug_idx, aug_tile in enumerate(augmented_tiles):
                # Generate unique ID for this tile
                tile_id = f"{image_name}_{tile_idx}_{aug_idx}_{uuid.uuid4().hex[:8]}"
                
                # Decide if this should be a clean-clean pair (5-10% of tiles)
                is_clean_pair = np.random.random() < 0.07  # 7% chance for clean pairs
                
                if is_clean_pair:
                    # For clean-clean pairs: input = output (no correction needed)
                    # Apply light denoising to both ground truth and "blurred" version
                    clean_gt = denoise_image(aug_tile, strength=denoise_strength * 0.5)  # Light denoising
                    blurred_versions = [clean_gt.copy()]  # Input same as output
                    print(f"Created clean-clean pair for tile {tile_id}")
                else:
                    # Normal processing with proper GT denoising
                    blurred_versions, clean_gt = create_blurred_versions(
                        aug_tile, 
                        normalize_back=True, 
                        orig_range=orig_range,
                        add_noise=True,
                        noise_preservation=noise_preservation,
                        edge_aware=edge_aware,
                        beta=beta,
                        denoise_ground_truth=True,
                        gt_denoise_strength=denoise_strength
                    )
                
                # Save the tile set
                save_paths = save_tile_set(
                    clean_gt,  # Now using the properly denoised ground truth
                    blurred_versions, 
                    tile_id, 
                    output_dirs, 
                    orig_range,
                    is_clean_pair=is_clean_pair
                )
                
                # Include additional metadata for balanced dataset tracking
                meta_info = {}
                for key in ['feature_type', 'contrast_band', 'brightness_band', 'spatial_region']:
                    if key in tile_info:
                        meta_info[key] = tile_info[key]
                
                # Return tile mapping info
                result_tiles.append({
                    'tile_id': tile_id,
                    'source_image': image_path,
                    'position': tile_info['position'],
                    'original_size': tile_info['original_size'],
                    'metrics': {k: float(v) if isinstance(v, (int, float)) else v 
                            for k, v in tile_info['metrics'].items()},
                    'metadata': meta_info,
                    'paths': save_paths,
                    'augmentation_idx': aug_idx,
                    'is_validation': (image_path in validation_images),
                    'is_clean_pair': is_clean_pair,  # Track clean pairs
                    'gt_was_denoised': True  # Always true now
                })
            
            return result_tiles
        
        # Use multithreading for tile processing
        all_results = []
        if len(accepted_tiles) > 0:
            print(f"Processing {len(accepted_tiles)} tiles with {args.augmentations} augmentations each...")
            
            # Create a partial function with fixed arguments
            process_tile_partial = partial(
                process_tile, 
                augmentations=args.augmentations, 
                tile_size=args.tile_size, 
                output_dirs=output_dirs,
                orig_range=orig_range,
                image_name=image_name,
                denoise_strength=args.denoise_strength,
                noise_preservation=args.noise_preservation,
                edge_aware=args.edge_aware_blur,
                beta=args.moffat_beta
            )
            
            # Process tiles in parallel using ThreadPoolExecutor
            with ThreadPoolExecutor(max_workers=min(16, os.cpu_count() or 4)) as executor:  # Increased workers
                futures = [executor.submit(process_tile_partial, tile_info, tile_idx) 
                        for tile_idx, tile_info in enumerate(accepted_tiles)]
                
                for future in tqdm(futures, desc="Creating tiles & augmentations", mininterval=0.1):
                    results = future.result()
                    all_results.extend(results)
                    total_tiles_generated += len(results)
        
        # Update tile mapping
        for result in all_results:
            tile_id = result.pop('tile_id')
            tile_mapping[tile_id] = result

        # Save mapping after each image (in case of interruption)
        if not args.estimate_only:
            # Save unified mapping file
            with open(os.path.join(args.output_dir, 'tile_mapping.json'), 'w') as f:
                json.dump(tile_mapping, f, indent=2)
    
    # Calculate estimated total tiles
    num_blur_versions = 4  # For 'Blur 1', 'Blur 2', 'Blur 3', 'Blur 4'
    total_estimated_tiles = total_accepted_tiles * (args.augmentations + 1) * (num_blur_versions + 1)  # +1 for ground truth

    if args.estimate_only:
        print(f"\nEstimated dataset will create {total_estimated_tiles} total tiles from the training data")
        print(f"- {total_accepted_tiles} tiles will be extracted and accepted")
        print(f"- Each tile creates {args.augmentations + 1} versions (original + {args.augmentations} augmentations)")
        print(f"- Each augmented tile produces 5 images (ground truth + 4 blur levels)")
        return {}
    else:
        # Count validation vs training tiles
        train_count = sum(1 for info in tile_mapping.values() if not info.get('is_validation', False))
        val_count = sum(1 for info in tile_mapping.values() if info.get('is_validation', False))
        
        print(f"\nDataset generation complete!")
        print(f"Generated {total_tiles_generated} total tiles from {len(image_paths)} images")
        print(f" - Training: {train_count} tiles")
        print(f" - Validation: {val_count} tiles (tracked in metadata)")
        print(f"Extracted {total_background_patches} background patches")
        print(f"Mapping saved to: {os.path.join(args.output_dir, 'tile_mapping.json')}")
        return tile_mapping
    
if __name__ == "__main__":
    args = parse_args()
    
    if args.estimate_only:
        print("\nEstimating dataset size only - no files will be created")
        generate_dataset(args)
        print("\nEstimation complete. Re-run without --estimate-only to create the dataset.")
    else:
        # Ask for confirmation before proceeding
        proceed = input(f"\nThis will create a training dataset in {args.output_dir}. Continue? (y/n): ")
        if proceed.lower() != 'y':
            print("Operation cancelled.")
            sys.exit(0)
        
        generate_dataset(args)
