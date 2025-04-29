import numpy as np
import matplotlib.pyplot as plt
import cv2
from skimage import exposure

def load_image(path, grayscale=True):
    """Load an image from path and convert to grayscale if specified."""
    if grayscale:
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    else:
        img = cv2.imread(path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert from BGR to RGB
    
    if img is None:
        raise ValueError(f"Could not load image from {path}")
    return img

def display_images(images, titles, figsize=(15, 10), cmaps=None, histograms=False):
    """Display multiple images with titles."""
    n = len(images)
    if histograms:
        fig, axs = plt.subplots(2, n, figsize=figsize)
        
        for i, (img, title) in enumerate(zip(images, titles)):
            if cmaps and i < len(cmaps):
                axs[0, i].imshow(img, cmap=cmaps[i])
            else:
                axs[0, i].imshow(img, cmap='gray')
            axs[0, i].set_title(title)
            axs[0, i].axis('off')
            
            # Plot histogram
            hist, bins = np.histogram(img.flatten(), 256, [0, 256])
            axs[1, i].bar(bins[:-1], hist, width=1, color='gray')
            axs[1, i].set_xlim([0, 256])
            axs[1, i].set_title(f'Histogram: {title}')
            
    else:
        fig, axs = plt.subplots(1, n, figsize=figsize)
        if n == 1:
            axs = [axs]
            
        for i, (img, title) in enumerate(zip(images, titles)):
            if cmaps and i < len(cmaps):
                axs[i].imshow(img, cmap=cmaps[i])
            else:
                axs[i].imshow(img, cmap='gray')
            axs[i].set_title(title)
            axs[i].axis('off')
    
    plt.tight_layout()
    return fig

# Topic 1: Log Transformation
def log_transform(image, c=1):
    """Apply log transformation to enhance image contrast."""
    # Avoid log(0) by adding a small constant
    float_img = image.astype(float) + 1e-10
    # Apply log transformation
    log_img = c * np.log(1 + float_img)
    # Normalize to [0, 255] range
    normalized = (log_img - np.min(log_img)) / (np.max(log_img) - np.min(log_img)) * 255
    return normalized.astype(np.uint8)

def demonstrate_log_transform(image_path):
    """Demonstrate log transformation with different constants."""
    img = load_image(image_path)
    
    log_c1 = log_transform(img, c=1)
    log_c5 = log_transform(img, c=5)
    log_c20 = log_transform(img, c=20)
    
    fig = display_images(
        [img, log_c1, log_c5, log_c20],
        ['Original', 'Log (c=1)', 'Log (c=5)', 'Log (c=20)']
    )
    
    return fig

# Topic 2: Gamma Transformation
def gamma_transform(image, gamma):
    """Apply gamma correction to the image."""
    # Normalize to [0, 1] range
    normalized = image / 255.0
    # Apply gamma correction
    corrected = np.power(normalized, gamma)
    # Convert back to [0, 255] range
    corrected = (corrected * 255).astype(np.uint8)
    return corrected

def demonstrate_gamma_transform(image_path):
    """Demonstrate gamma transformation with different gamma values."""
    img = load_image(image_path)
    
    gamma_0_5 = gamma_transform(img, 0.5)
    gamma_1_0 = gamma_transform(img, 1.0)  # Should be same as original
    gamma_2_0 = gamma_transform(img, 2.0)
    gamma_3_0 = gamma_transform(img, 3.0)
    
    fig = display_images(
        [img, gamma_0_5, gamma_1_0, gamma_2_0, gamma_3_0],
        ['Original', 'Gamma=0.5', 'Gamma=1.0', 'Gamma=2.0', 'Gamma=3.0']
    )
    
    return fig

# Topic 3: Piecewise Linear Transformation
def piecewise_linear_transform(image, r1, r2, s1=0, s2=255):
    """
    Apply piecewise linear transformation.
    Maps values in range [r1, r2] to range [s1, s2].
    """
    result = np.copy(image).astype(float)
    
    # Values below r1
    result[image < r1] = s1 * (image[image < r1] / r1)
    
    # Values between r1 and r2
    mask = (image >= r1) & (image <= r2)
    result[mask] = s1 + (s2 - s1) * ((image[mask] - r1) / (r2 - r1))
    
    # Values above r2
    result[image > r2] = s2 + (255 - s2) * ((image[image > r2] - r2) / (255 - r2))
    
    return np.clip(result, 0, 255).astype(np.uint8)

def demonstrate_piecewise_linear_transform(image_path):
    """Demonstrate piecewise linear transformation with different ranges."""
    img = load_image(image_path)
    
    # Stretch [50, 150] to [0, 255]
    transform_50_150 = piecewise_linear_transform(img, 50, 150)
    
    # Stretch [100, 200] to [0, 255]
    transform_100_200 = piecewise_linear_transform(img, 100, 200)
    
    # Stretch [0, 128] to [0, 255]
    transform_0_128 = piecewise_linear_transform(img, 0, 128)
    
    fig = display_images(
        [img, transform_50_150, transform_100_200, transform_0_128],
        ['Original', '[50, 150] -> [0, 255]', '[100, 200] -> [0, 255]', '[0, 128] -> [0, 255]']
    )
    
    return fig

# Topic 4: Bit Plane Slicing
def extract_bit_plane(image, bit_pos):
    """Extract the specified bit plane from the image."""
    # Create a mask for the bit position
    mask = 1 << bit_pos
    # Extract the bit plane using bitwise AND
    bit_plane = (image & mask) >> bit_pos
    # Scale to 0 or 255 for better visibility
    bit_plane = bit_plane * 255
    return bit_plane.astype(np.uint8)

def demonstrate_bit_plane_slicing(image_path):
    """Demonstrate bit plane slicing for different bit positions."""
    img = load_image(image_path)
    
    # Extract bit planes from MSB (7) to LSB (0)
    bit_planes = [extract_bit_plane(img, i) for i in range(8)]
    
    # First figure: Original and MSB (7, 6, 5, 4)
    fig1 = display_images(
        [img, bit_planes[7], bit_planes[6], bit_planes[5], bit_planes[4]],
        ['Original', 'Bit 7 (MSB)', 'Bit 6', 'Bit 5', 'Bit 4']
    )
    
    # Second figure: LSB (3, 2, 1, 0)
    fig2 = display_images(
        [bit_planes[3], bit_planes[2], bit_planes[1], bit_planes[0]],
        ['Bit 3', 'Bit 2', 'Bit 1', 'Bit 0 (LSB)']
    )
    
    return fig1, fig2

# Topic 5: Histogram Equalization
def histogram_equalization(image):
    """Apply histogram equalization to the image."""
    return cv2.equalizeHist(image)

def demonstrate_histogram_equalization(image_path):
    """Demonstrate histogram equalization and plot histograms."""
    img = load_image(image_path)
    
    # Apply histogram equalization
    equalized = histogram_equalization(img)
    
    # Display images and histograms
    fig = display_images(
        [img, equalized],
        ['Original', 'Histogram Equalized'],
        histograms=True
    )
    
    return fig

# Additional Exploration: Combined Transformations
def demonstrate_combined_transformations(image_path):
    """Demonstrate combined transformations: log followed by gamma."""
    img = load_image(image_path)
    
    # Apply log transformation followed by gamma correction
    log_img = log_transform(img, c=5)
    log_gamma_img = gamma_transform(log_img, 0.7)
    
    # Apply gamma correction followed by log transformation
    gamma_img = gamma_transform(img, 0.5)
    gamma_log_img = log_transform(gamma_img, c=5)
    
    fig = display_images(
        [img, log_img, log_gamma_img, gamma_img, gamma_log_img],
        ['Original', 'Log (c=5)', 'Log + Gamma (0.7)', 'Gamma (0.5)', 'Gamma + Log (c=5)']
    )
    
    return fig

# Main function to run all demonstrations
def main():
    # You'll need to provide an image path here
    image_path = 'Grayscale_8bits_palette_sample_image.png'  # Replace with your image path
    
    print("1. Log Transformation")
    fig1 = demonstrate_log_transform(image_path)
    fig1.savefig('log_transformation.png')
    
    print("2. Gamma Transformation")
    fig2 = demonstrate_gamma_transform(image_path)
    fig2.savefig('gamma_transformation.png')
    
    print("3. Piecewise Linear Transformation")
    fig3 = demonstrate_piecewise_linear_transform(image_path)
    fig3.savefig('piecewise_linear_transformation.png')
    
    print("4. Bit Plane Slicing")
    fig4_1, fig4_2 = demonstrate_bit_plane_slicing(image_path)
    fig4_1.savefig('bit_plane_slicing_msb.png')
    fig4_2.savefig('bit_plane_slicing_lsb.png')
    
    print("5. Histogram Equalization")
    fig5 = demonstrate_histogram_equalization(image_path)
    fig5.savefig('histogram_equalization.png')
    
    print("6. Combined Transformations")
    fig6 = demonstrate_combined_transformations(image_path)
    fig6.savefig('combined_transformations.png')
    
    plt.show()

if __name__ == "__main__":
    main()