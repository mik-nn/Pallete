import colour
import numpy as np
from numba import jit
from sklearn.cluster import KMeans, MeanShift
from sklearn.mixture import GaussianMixture
import matplotlib.pyplot as plt
from PIL import Image
from IPython.display import display
from timex import timex

# Vectorized implementation of RGB to OKLab conversion
def rgb_to_oklab(rgb):
    rgb = rgb / 255.0
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
    return rgb.astype(int)

# Convert OKLab to cylindrical representation (L, C, h)
def oklab_to_cylindrical(oklab):
    L, a, b = oklab
    C = np.sqrt(a**2 + b**2)
    h = np.arctan2(b, a)
    CB = np.sqrt(L**2+C**2)
    return np.array([L, C, h, CB])

# Convert cylindrical representation (L, C, h) back to OKLab
def cylindrical_to_oklab(cylindrical):
    L, C, h = cylindrical
    a = C * np.cos(h)
    b = C * np.sin(h)
    return np.array([L, a, b])    

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
    
    # Use KMeans to cluster the OKLab pixels
    kmeans = KMeans(n_clusters=num_colors, random_state=42, n_init=10)
    kmeans.fit(oklab_pixels)
    
    # Get the colors (cluster centers) in OKLab
    oklab_colors = kmeans.cluster_centers_
    
    # Convert OKLab colors to cylindrical representation
    cylindrical_colors = np.array([oklab_to_cylindrical(color) for color in oklab_colors])
    
    # Sort the colors first by L (lightness) and then by h (hue)
    cylindrical_colors = sorted(cylindrical_colors, key=lambda color: (color[0], color[2]), reverse=True)
    
    # Convert cylindrical colors back to OKLab
    sorted_oklab_colors = np.array([cylindrical_to_oklab(color) for color in cylindrical_colors])
    
    # Convert OKLab colors back to RGB
    rgb_colors = oklab_to_rgb(sorted_oklab_colors)

    return rgb_colors

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

if __name__ == "__main__":
    image_paths = [
        'anna-pelzer-IGfIGP5ONV0-unsplash.jpg',
        'bird-8666099_640.jpg',
        'goose-8740266_640.jpg'
    ]
    
    num_colors = 25  # Number of colors in the palette

    for image_path in image_paths:
        if image_path.startswith('/home/mikz/PetProject/'):
            # For absolute paths, use as is
            full_path = image_path
        else:
            # For relative paths, assume they're in current directory
            full_path = image_path
        
        try:
            print(f"Processing {image_path}...")
            colors = extract_palette(full_path, num_colors)
            plot_image_with_palette(full_path, colors)
            print(f"Successfully processed {image_path}")
        except Exception as e:
            print(f"Error processing {image_path}: {e}")