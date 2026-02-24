#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from timex import timex
from sklearn.cluster import KMeans
from IPython.display import display

# Vectorized implementation of RGB to OKLab conversion

def rgb_to_oklab(rgb):
    rgb = rgb.astype(np.float32) / 255.0
    #print(rgb)
    l = 0.4122214708 * rgb[:, 0] + 0.5363325363 * rgb[:, 1] + 0.0514459929 * rgb[:, 2]
    m = 0.2119034982 * rgb[:, 0] + 0.6806995451 * rgb[:, 1] + 0.1073969566 * rgb[:, 2]
    s = 0.0883024619 * rgb[:, 0] + 0.2817188376 * rgb[:, 1] + 0.6299787005 * rgb[:, 2]

    l_ = np.cbrt(l)
    m_ = np.cbrt(m)
    s_ = np.cbrt(s)

    L = 0.2104542553 * l_ + 0.7936177850 * m_ - 0.0040720468 * s_
    a = 1.9779984951 * l_ - 2.4285922050 * m_ + 0.4505937099 * s_
    b = 0.0259040371 * l_ + 0.7827717662 * m_ - 0.8086757660 * s_

    return np.stack([L, a, b], axis=1)

# Vectorized implementation of OKLab to RGB conversion

def oklab_to_rgb(oklab):
    L, a, b = oklab[:, 0], oklab[:, 1], oklab[:, 2]

    l_ = L + 0.3963377774 * a + 0.2158037573 * b
    m_ = L - 0.1055613458 * a - 0.0638541728 * b
    s_ = L - 0.0894841775 * a - 1.2914855480 * b

    l = l_ ** 3
    m = m_ ** 3
    s = s_ ** 3

    r = 4.0767416621 * l - 3.3077115913 * m + 0.2309699292 * s
    g = -1.2684380046 * l + 2.6097574011 * m - 0.3413193965 * s
    b = -0.0041960863 * l - 0.7034186147 * m + 1.7076147010 * s

    rgb = np.clip(np.stack([r, g, b], axis=1), 0, 1) * 255
    return rgb.astype(np.uint8)

# Convert OKLab to cylindrical representation (L, C, h)

def oklab_to_cylindrical(oklab):
    L, a, b = oklab[0], oklab[1], oklab[2]
    C = np.sqrt(a**2 + b**2)
    h = np.arctan2(b, a)
    return np.array([L, C, h])

# Convert cylindrical representation (L, C, h) back to OKLab

def cylindrical_to_oklab(cylindrical):
    L, C, h = cylindrical
    a = C * np.cos(h)
    b = C * np.sin(h)
    return np.array([L, a, b])    

def color_distance(color1, color2):
    """Calculate the distance between two OKLab colors using cylindrical coordinates"""
    # Note: Vectorized version for performance
    if color1.ndim == 1 and color2.ndim == 1:
        L1, a1, b1 = color1
        L2, a2, b2 = color2
    elif color1.ndim == 1:
        L1, a1, b1 = color1
        L2, a2, b2 = color2[:, 0], color2[:, 1], color2[:, 2]
    elif color2.ndim == 1:
        L1, a1, b1 = color1[:, 0], color1[:, 1], color1[:, 2]
        L2, a2, b2 = color2
    else:
        L1, a1, b1 = color1[:, 0], color1[:, 1], color1[:, 2]
        L2, a2, b2 = color2[:, 0], color2[:, 1], color2[:, 2]
    
    # Calculate Cylindrical coordinates
    C1 = np.sqrt(a1**2 + b1**2)
    C2 = np.sqrt(a2**2 + b2**2)
    h1 = np.arctan2(b1, a1)
    h2 = np.arctan2(b2, a2)
    
    # Distance in cylindrical space (L, C, h)
    L_diff = L1 - L2
    C_diff = C1 - C2
    
    h_diff = np.abs(h1 - h2)
    h_diff = np.minimum(h_diff, 2*np.pi - h_diff)  # Handle hue wrap-around
    
    return np.sqrt(L_diff**2 + C_diff**2 + h_diff**2)

def select_furthest_colors(oklab_pixels, num_colors):
    """Select colors that are furthest apart from each other"""
    # Subsample pixels if there are too many to speed up initial search
    # 100x100 = 10000 pixels. A subset of 1000 is usually enough for a good palette distribution
    max_sample_size = 1000
    if len(oklab_pixels) > max_sample_size:
        indices = np.random.choice(len(oklab_pixels), max_sample_size, replace=False)
        sampled_pixels = oklab_pixels[indices]
    else:
        sampled_pixels = oklab_pixels

    if len(sampled_pixels) <= num_colors:
        return sampled_pixels
    
    # Start with the color that has the most average distance to all other colors
    # Vectorized distance calculation to find the first color
    total_distances = np.zeros(len(sampled_pixels))
    for i, color in enumerate(sampled_pixels):
        total_distances[i] = np.sum(color_distance(color, sampled_pixels))
        
    first_color_idx = np.argmax(total_distances)
    selected_colors = [sampled_pixels[first_color_idx]]
    
    # Iteratively select the color that is furthest from all already selected colors
    while len(selected_colors) < num_colors:
        # Calculate distances from all candidates to all selected colors
        # Shape: (num_selected, num_candidates)
        all_distances = np.array([color_distance(sc, sampled_pixels) for sc in selected_colors])
        
        # For each candidate, find the minimum distance to ANY selected color
        # Shape: (num_candidates,)
        min_distances = np.min(all_distances, axis=0)
        
        # To avoid picking exactly the same color again, set distance of already selected to -1
        for sc in selected_colors:
            # Find matching pixels. np.all with axis=1 checks if all RGB components match
            matches = np.all(sampled_pixels == sc, axis=1)
            min_distances[matches] = -1
            
        # Pick the candidate that maximizes this minimum distance
        next_color_idx = np.argmax(min_distances)
        
        if min_distances[next_color_idx] > 0:
            selected_colors.append(sampled_pixels[next_color_idx])
        else:
            break # All colors perfectly overlap
    
    return np.array(selected_colors)

@timex
def extract_palette(image_path, num_colors):
    # Open the image
    image = Image.open(image_path)
    image = image.convert('RGB')  # Ensure image is in RGB format

    # Resize image to reduce the number of pixels for faster processing
    image = image.resize((100, 100))

    # Convert image to numpy array
    image_array = np.array(image)

    # Reshape the image array to be a list of pixels
    pixels = image_array.reshape(-1, 3)

    # Convert RGB pixels to OKLab
    oklab_pixels = rgb_to_oklab(pixels)
    
    # Select colors that are furthest apart
    selected_oklab_colors = select_furthest_colors(oklab_pixels, num_colors)
    
    # Convert selected OKLab colors to cylindrical representation for sorting
    cylindrical_colors = np.array([oklab_to_cylindrical(color) for color in selected_oklab_colors])
    
    # Sort the colors by L (lightness) and then by h (hue)
    cylindrical_colors = sorted(cylindrical_colors, key=lambda color: (color[0], color[2]), reverse=True)
    
    # Convert cylindrical colors back to OKLab
    sorted_oklab_colors = np.array([cylindrical_to_oklab(color) for color in cylindrical_colors])

    # Convert OKLab colors back to RGB
    rgb_colors = oklab_to_rgb(sorted_oklab_colors)

    return rgb_colors, oklab_pixels

def plot_palette(colors):
    # Create a figure and a set of subplots
    fig, ax = plt.subplots(1, 1, figsize=(8, 2), subplot_kw=dict(xticks=[], yticks=[], frame_on=False))

    # Create a color bar
    for i, color in enumerate(colors):
        ax.add_patch(plt.Rectangle((i, 0), 1, 1, color=np.array(color) / 255.0))

    plt.xlim(0, len(colors))
    plt.ylim(0, 1)
    plt.show()

def plot_image_with_palette(image_path, colors):
    # Open the image
    image = Image.open(image_path)

    # Create a figure and a set of subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6), gridspec_kw={'width_ratios': [8, 1]})
    fig.suptitle(image_path, fontsize=16)

    # Display the image
    ax1.imshow(image)
    ax1.axis('off')

    # Create a color bar for the palette
    for i, color in enumerate(colors):
        ax2.add_patch(plt.Rectangle((0, i), 1, 1, color=np.array(color) / 255.0))

    ax2.set_xlim(0, 1)
    ax2.set_ylim(0, len(colors))
    ax2.axis('off')

    plt.show()

def plot_ab_projection(image_path, selected_colors, oklab_pixels=None):
    """Plot pixels projected to a-b space with selected colors highlighted"""
    if oklab_pixels is None:
        # Load and process image to get OKLab pixels
        image = Image.open(image_path)
        image = image.convert('RGB')
        image = image.resize((100, 100))
        image_array = np.array(image)
        pixels = image_array.reshape(-1, 3)
        oklab_pixels = rgb_to_oklab(pixels)
    
    # Extract a and b components
    a_values = oklab_pixels[:, 1]
    b_values = oklab_pixels[:, 2]
    
    # Convert selected colors to OKLab for projection
    selected_oklab = []
    for rgb_color in selected_colors:
        # Convert RGB to OKLab
        rgb_array = np.array(rgb_color).reshape(1, 3).astype(np.float32)
        oklab_color = rgb_to_oklab(rgb_array)[0]
        selected_oklab.append(oklab_color)
    
    selected_oklab = np.array(selected_oklab)
    selected_a = selected_oklab[:, 1]
    selected_b = selected_oklab[:, 2]
    
    # Create the plot with better layout
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
    
    # Plot 1: All pixels with density
    ax1.scatter(a_values, b_values, c='lightblue', alpha=0.2, s=2, label='All pixels')
    ax1.scatter(selected_a, selected_b, c='red', s=150, alpha=0.9, 
                label='Selected colors', edgecolors='black', linewidth=2, zorder=5)
    
    # Add color labels for selected colors
    for i, (a, b) in enumerate(zip(selected_a, selected_b)):
        ax1.annotate(f'{i+1}', (a, b), xytext=(8, 8), textcoords='offset points',
                    fontsize=10, fontweight='bold', color='black',
                    bbox=dict(boxstyle='circle,pad=0.3', facecolor='white', alpha=0.7))
    
    ax1.set_xlabel('a (green-red axis)', fontsize=12)
    ax1.set_ylabel('b (blue-yellow axis)', fontsize=12)
    ax1.set_title(f'Color Distribution in a-b Space - {image_path}', fontsize=14)
    ax1.grid(True, alpha=0.3)
    ax1.legend(fontsize=10)
    ax1.axis('equal')
    
    # Add color circle for reference
    theta = np.linspace(0, 2*np.pi, 100)
    for radius in [0.2, 0.4, 0.6, 0.8]:
        circle_a = radius * np.cos(theta)
        circle_b = radius * np.sin(theta)
        ax1.plot(circle_a, circle_b, 'k--', alpha=0.3, linewidth=0.5)
    
    # Plot 2: Selected colors with actual color representation
    for i, (a, b) in enumerate(zip(selected_a, selected_b)):
        # Create a small square with the actual color
        color_rgb = np.array(selected_colors[i]) / 255.0
        ax2.add_patch(plt.Rectangle((i-0.4, -0.1), 0.8, 0.2, 
                                 facecolor=color_rgb, edgecolor='black', linewidth=1))
        # Add position marker on AB diagram
        ax2.scatter(a, b, c=[color_rgb], s=100, alpha=0.8, edgecolors='black', linewidth=1)
        ax2.annotate(f'{i+1}', (a, b), xytext=(5, 5), textcoords='offset points',
                    fontsize=8, fontweight='bold', color='black')
    
    ax2.set_xlabel('a (green-red axis)', fontsize=12)
    ax2.set_ylabel('b (blue-yellow axis)', fontsize=12)
    ax2.set_title('Selected Colors in a-b Space', fontsize=14)
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(-1, 1)
    ax2.set_ylim(-1, 1)
    ax2.axis('equal')
    
    # Add color circle for reference
    for radius in [0.2, 0.4, 0.6, 0.8]:
        circle_a = radius * np.cos(theta)
        circle_b = radius * np.sin(theta)
        ax2.plot(circle_a, circle_b, 'k--', alpha=0.3, linewidth=0.5)
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    image_paths = [
        'anna-pelzer-IGfIGP5ONV0-unsplash.jpg',
        'bird-8666099_640.jpg',
        'goose-8740266_640.jpg',
        'wild-goose-2260866_640.jpg'
    ]
    
    num_colors = 25  # Number of colors in the palette

    for image_path in image_paths:
        try:
            print(f"Processing {image_path}...")
            
            # Load and process image to get OKLab pixels for projection
            image = Image.open(image_path)
            image = image.convert('RGB')
            image = image.resize((100, 100))
            image_array = np.array(image)
            pixels = image_array.reshape(-1, 3)
            oklab_pixels = rgb_to_oklab(pixels)
            
            # Extract palette
            colors, _ = extract_palette(image_path, num_colors)
            
            # Plot image with palette
            plot_image_with_palette(image_path, colors)
            
            # Plot a-b projection
            plot_ab_projection(image_path, colors, oklab_pixels)
            
            print(f"Successfully processed {image_path}")
        except Exception as e:
            print(f"Error processing {image_path}: {e}")

