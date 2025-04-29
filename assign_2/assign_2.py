import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage
import random
import os

def add_salt_pepper_noise(image, prob=0.2):
    """
    Add salt and pepper noise to an image.
    
    Parameters:
    image (ndarray): Input image
    prob (float): Probability of noise (default 0.2 for 1/5 of pixels)
    
    Returns:
    ndarray: Noisy image
    """
    noisy_img = np.copy(image)
    h, w = image.shape
    
    # Calculate number of pixels to change (1/5 of total pixels)
    num_pixels = int(prob * h * w)
    
    # Add salt (white) noise
    for _ in range(num_pixels // 2):
        y = random.randint(0, h - 1)
        x = random.randint(0, w - 1)
        noisy_img[y, x] = 255
    
    # Add pepper (black) noise
    for _ in range(num_pixels // 2):
        y = random.randint(0, h - 1)
        x = random.randint(0, w - 1)
        noisy_img[y, x] = 0
        
    return noisy_img

def apply_filter_correlation(image, kernel):
    """
    Apply correlation between an image and a kernel.
    
    Parameters:
    image (ndarray): Input image
    kernel (ndarray): Filter kernel
    
    Returns:
    ndarray: Filtered image
    """
    return ndimage.correlate(image, kernel, mode='reflect')

def apply_filter_convolution(image, kernel):
    """
    Apply convolution between an image and a kernel.
    
    Parameters:
    image (ndarray): Input image
    kernel (ndarray): Filter kernel
    
    Returns:
    ndarray: Filtered image
    """
    # For convolution, we flip the kernel
    flipped_kernel = np.flipud(np.fliplr(kernel))
    return ndimage.correlate(image, flipped_kernel, mode='reflect')

def create_gaussian_kernel(size, sigma):
    """
    Create a Gaussian kernel of specified size and standard deviation.
    
    Parameters:
    size (int): Size of the kernel (should be odd)
    sigma (float): Standard deviation
    
    Returns:
    ndarray: Gaussian kernel
    """
    # Ensure size is odd
    if size % 2 == 0:
        size += 1
    
    # Create 1D coordinate arrays
    ax = np.linspace(-(size - 1) / 2, (size - 1) / 2, size)
    xx, yy = np.meshgrid(ax, ax)
    
    # Compute Gaussian values
    kernel = np.exp(-(xx**2 + yy**2) / (2 * sigma**2))
    
    # Normalize kernel
    return kernel / np.sum(kernel)

def apply_median_filter(image, size):
    """
    Apply median filter to an image.
    
    Parameters:
    image (ndarray): Input image
    size (int): Size of the kernel
    
    Returns:
    ndarray: Filtered image
    """
    return ndimage.median_filter(image, size=size)

def calculate_gradient_sobel(image):
    """
    Calculate gradient using Sobel filters.
    
    Parameters:
    image (ndarray): Input image
    
    Returns:
    tuple: (gradient_magnitude, gradient_direction, gradient_x, gradient_y)
    """
    # Use OpenCV's built-in Sobel implementation which is more robust
    gradient_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
    gradient_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)
    
    # Calculate gradient magnitude and direction
    gradient_magnitude = np.sqrt(gradient_x**2 + gradient_y**2)
    
    # Handle NaN or Inf values that can cause normalization to fail
    gradient_magnitude = np.nan_to_num(gradient_magnitude)
    
    # Normalize to 0-255 range for display
    gradient_magnitude = cv2.normalize(gradient_magnitude, None, 0, 255, cv2.NORM_MINMAX)
    gradient_magnitude = gradient_magnitude.astype(np.uint8)
    
    # Calculate direction (in degrees)
    gradient_direction = np.arctan2(gradient_y, gradient_x) * (180 / np.pi)
    
    return gradient_magnitude, gradient_direction, gradient_x, gradient_y

def apply_laplacian_filter(image):
    """
    Apply Laplacian filter to an image.
    
    Parameters:
    image (ndarray): Input image
    
    Returns:
    ndarray: Filtered image
    """
    # Define Laplacian kernel
    laplacian_kernel = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=np.float32)
    
    # Apply Laplacian filter
    laplacian = apply_filter_convolution(image, laplacian_kernel)
    
    # Convert to 8-bit format for display
    laplacian = np.nan_to_num(laplacian)  # Handle any NaN values
    return cv2.normalize(laplacian, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

def apply_log_filter(image, sigma):
    """
    Apply Laplacian of Gaussian (LoG) filter to an image.
    
    Parameters:
    image (ndarray): Input image
    sigma (float): Standard deviation for Gaussian
    
    Returns:
    ndarray: Filtered image
    """
    # First apply Gaussian blur
    gaussian = ndimage.gaussian_filter(image, sigma=sigma)
    
    # Then apply Laplacian
    laplacian_kernel = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=np.float32)
    log = apply_filter_convolution(gaussian, laplacian_kernel)
    
    # Convert to 8-bit format for display
    log = np.nan_to_num(log)  # Handle any NaN values
    return cv2.normalize(log, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

def unsharp_masking(image, sigma=3, k=1):
    """
    Apply unsharp masking to an image.
    
    Parameters:
    image (ndarray): Input image
    sigma (float): Standard deviation for Gaussian blur
    k (float): Weight factor for the mask
    
    Returns:
    ndarray: Sharpened image
    """
    # Step 1: Gaussian blur
    blurred = ndimage.gaussian_filter(image, sigma=sigma)
    
    # Step 2: Create mask by subtracting blurred from original
    mask = image - blurred
    
    # Step 3: Add weighted mask to original
    sharpened = image + k * mask
    
    # Clip values to valid range [0, 255]
    return np.clip(sharpened, 0, 255).astype(np.uint8)

def highboost_filtering(image, sigma=3, k=2):
    """
    Apply highboost filtering to an image.
    
    Parameters:
    image (ndarray): Input image
    sigma (float): Standard deviation for Gaussian blur
    k (float): Boost factor
    
    Returns:
    ndarray: Highboost filtered image
    """
    # This is the same as unsharp masking but with a higher k value
    return unsharp_masking(image, sigma, k)

def save_and_display_images(images_dict, filename_prefix, figsize=(15, 10)):
    """
    Save and display multiple images.
    
    Parameters:
    images_dict (dict): Dictionary of {title: image}
    filename_prefix (str): Prefix for saved files
    figsize (tuple): Figure size
    """
    n = len(images_dict)
    cols = min(3, n)
    rows = (n + cols - 1) // cols
    
    plt.figure(figsize=figsize)
    
    for i, (title, img) in enumerate(images_dict.items()):
        plt.subplot(rows, cols, i + 1)
        if len(img.shape) == 2 or img.shape[2] == 1:
            plt.imshow(img, cmap='gray')
        else:
            plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        plt.title(title)
        plt.axis('off')
        
        # Save individual image
        cv2.imwrite(f"{filename_prefix}_{title.replace(' ', '_')}.png", img)
    
    plt.tight_layout()
    plt.savefig(f"{filename_prefix}_comparison.png")
    
    # Don't use plt.show() in non-interactive environments
    # plt.show()  # This is causing the warning
    plt.close()  # Close the figure instead

def main():
    # Create a directory for output images if it doesn't exist
    if not os.path.exists('output'):
        os.makedirs('output')
    
    # Fixed image path
    image_path = "Grayscale_8bits_palette_sample_image.png"
    
    print(f"Attempting to load image from: {image_path}")
    original_image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    
    if original_image is None:
        print(f"Error: Could not read image from {image_path}")
        print("Please make sure the image file exists in the current directory.")
        print(f"Current working directory: {os.getcwd()}")
        print("Files in current directory:")
        print(os.listdir('.'))
        return
    
    print(f"Successfully loaded image of shape: {original_image.shape}")
    
    # Topic 1: Spatial Filtering
    # Add salt-and-pepper noise
    noisy_image = add_salt_pepper_noise(original_image, prob=0.2)
    
    # Create 3x3 mean filter
    mean_kernel = np.ones((3, 3), dtype=np.float32) / 9
    
    # Apply mean filter using correlation and convolution
    mean_filtered_corr = apply_filter_correlation(noisy_image, mean_kernel)
    mean_filtered_conv = apply_filter_convolution(noisy_image, mean_kernel)
    
    # Display images for Topic 1
    save_and_display_images({
        'Original': original_image,
        'Noisy': noisy_image,
        'Mean Filter (Correlation)': mean_filtered_corr.astype(np.uint8),
        'Mean Filter (Convolution)': mean_filtered_conv.astype(np.uint8)
    }, 'output/topic1')
    
    # Topic 2: Gaussian Filter
    # Create Gaussian kernels
    gaussian_kernel_3x3 = create_gaussian_kernel(3, 3)
    gaussian_kernel_5x5 = create_gaussian_kernel(5, 3)
    
    # Apply Gaussian filters
    gaussian_filtered_3x3 = apply_filter_convolution(original_image, gaussian_kernel_3x3)
    gaussian_filtered_5x5 = apply_filter_convolution(original_image, gaussian_kernel_5x5)
    
    # Display images for Topic 2
    save_and_display_images({
        'Original': original_image,
        'Gaussian 3x3': gaussian_filtered_3x3.astype(np.uint8),
        'Gaussian 5x5': gaussian_filtered_5x5.astype(np.uint8)
    }, 'output/topic2')
    
    # Topic 3: Median Filter
    # Apply median filter
    median_filtered = apply_median_filter(noisy_image, size=3)
    
    # Display images for Topic 3
    save_and_display_images({
        'Noisy': noisy_image,
        'Median Filtered': median_filtered
    }, 'output/topic3')
    
    # Topic 4: Gradient Calculation and Sobel Filter
    try:
        # Calculate gradient using Sobel
        gradient_magnitude, gradient_direction, gradient_x, gradient_y = calculate_gradient_sobel(original_image)
        
        # Display results for Topic 4
        gradient_direction_display = ((gradient_direction + 180) % 360) / 360 * 255
        save_and_display_images({
            'Original': original_image,
            'Gradient Magnitude': gradient_magnitude,
            'Gradient Direction': gradient_direction_display.astype(np.uint8),
            'Gradient X': cv2.normalize(np.nan_to_num(gradient_x), None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8),
            'Gradient Y': cv2.normalize(np.nan_to_num(gradient_y), None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        }, 'output/topic4')
    except Exception as e:
        print(f"Error in Topic 4: {e}")
    
    # Topic 5: Laplacian Filter
    try:
        laplacian_filtered = apply_laplacian_filter(original_image)
        
        # Display results for Topic 5
        save_and_display_images({
            'Original': original_image,
            'Laplacian Filtered': laplacian_filtered
        }, 'output/topic5')
    except Exception as e:
        print(f"Error in Topic 5: {e}")
    
    # Topic 6: Laplacian of Gaussian (LoG)
    try:
        gaussian_filtered = ndimage.gaussian_filter(original_image, sigma=3)
        log_filtered = apply_log_filter(original_image, sigma=3)
        
        # Display results for Topic 6
        save_and_display_images({
            'Original': original_image,
            'Gaussian (σ=3)': gaussian_filtered.astype(np.uint8),
            'Laplacian': laplacian_filtered,
            'LoG (σ=3)': log_filtered
        }, 'output/topic6')
    except Exception as e:
        print(f"Error in Topic 6: {e}")
    
    # Topic 7: Unsharp Masking and Highboost Filtering
    try:
        unsharp_masked = unsharp_masking(original_image, sigma=3, k=1)
        highboost_filtered = highboost_filtering(original_image, sigma=3, k=2)
        
        # Display results for Topic 7
        save_and_display_images({
            'Original': original_image,
            'Unsharp Masked': unsharp_masked,
            'Highboost Filtered': highboost_filtered
        }, 'output/topic7')
    except Exception as e:
        print(f"Error in Topic 7: {e}")
    
    # Topic 8: Combining Filters
    try:
        # Example 1: Gaussian followed by Laplacian
        gaussian_then_laplacian = apply_laplacian_filter(gaussian_filtered.astype(np.uint8))
        
        # Example 2: Median followed by Sobel
        median_then_sobel, _, _, _ = calculate_gradient_sobel(median_filtered)
        
        # Display results for Topic 8
        save_and_display_images({
            'Original': original_image,
            'Gaussian then Laplacian': gaussian_then_laplacian,
            'Median then Sobel': median_then_sobel
        }, 'output/topic8')
    except Exception as e:
        print(f"Error in Topic 8: {e}")
    
    # Additional test on color image
    try:
        color_image = cv2.imread(image_path)
        if color_image is not None:
            # Process each channel separately
            b, g, r = cv2.split(color_image)
            b_filtered = apply_median_filter(b, size=3)
            g_filtered = apply_median_filter(g, size=3)
            r_filtered = apply_median_filter(r, size=3)
            color_filtered = cv2.merge([b_filtered, g_filtered, r_filtered])
            
            # Display color results
            save_and_display_images({
                'Original Color': color_image,
                'Median Filtered Color': color_filtered
            }, 'output/color_image')
        else:
            print("Warning: Could not load color image.")
    except Exception as e:
        print(f"Error in color image processing: {e}")
    
    print("All processing and saving complete!")

if __name__ == "__main__":
    main()