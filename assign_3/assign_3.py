import numpy as np
import cv2
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

def create_notch_filter(shape, d0, u0, v0):
    """
    Create a notch reject filter with center at (u0, v0)
    
    Parameters:
    - shape: Shape of the frequency domain image (height, width)
    - d0: Cutoff distance (radius of the notch)
    - u0, v0: Center of the notch filter in the frequency domain
    
    Returns:
    - Notch filter as a 2D array
    """
    rows, cols = shape
    x = np.arange(cols)
    y = np.arange(rows)
    x, y = np.meshgrid(x, y)
    
    # Calculate center-shifted coordinates
    x_shifted = x - cols // 2
    y_shifted = y - rows // 2
    
    # Calculate distance from notch center (u0, v0)
    d1 = np.sqrt((x_shifted - u0)**2 + (y_shifted - v0)**2)
    d2 = np.sqrt((x_shifted + u0)**2 + (y_shifted + v0)**2)
    
    # Create notch filter (Butterworth notch reject filter)
    n = 2  # Order of the Butterworth filter
    h = 1 / (1 + (d0**2 / (d1**2 + 1e-8))**n) * 1 / (1 + (d0**2 / (d2**2 + 1e-8))**n)
    
    return h

def apply_notch_filter(image_path, notch_centers, d0=10, display=True):
    """
    Apply notch filter to remove specific frequencies from an image
    
    Parameters:
    - image_path: Path to the input image
    - notch_centers: List of (u, v) coordinates for notch centers
    - d0: Radius of each notch
    - display: Whether to display intermediate results
    
    Returns:
    - Filtered image
    """
    # Read the image
    img = cv2.imread(image_path, 0)  # Read as grayscale
    if img is None:
        raise ValueError(f"Could not read image at {image_path}")
    
    # Transform to frequency domain
    f_transform = np.fft.fft2(img)
    f_shift = np.fft.fftshift(f_transform)
    
    # Initialize filter as all ones
    notch_filter = np.ones_like(f_shift, dtype=np.float32)
    
    # Apply each notch filter
    for u0, v0 in notch_centers:
        notch = create_notch_filter(img.shape, d0, u0, v0)
        notch_filter *= notch
    
    # Apply filter to the frequency domain
    filtered_f_shift = f_shift * notch_filter
    
    # Inverse FFT to get back to spatial domain
    f_ishift = np.fft.ifftshift(filtered_f_shift)
    img_back = np.fft.ifft2(f_ishift)
    img_filtered = np.abs(img_back)
    
    # Display results if requested
    if display:
        plt.figure(figsize=(15, 10))
        
        # Original image
        plt.subplot(2, 3, 1)
        plt.imshow(img, cmap='gray')
        plt.title('Original Image')
        
        # Magnitude spectrum (log scale for better visualization)
        plt.subplot(2, 3, 2)
        plt.imshow(np.log1p(np.abs(f_shift)), cmap='viridis')
        plt.title('Magnitude Spectrum (log scale)')
        
        # Notch filter
        plt.subplot(2, 3, 3)
        plt.imshow(notch_filter, cmap='gray')
        plt.title('Notch Filter')
        
        # Filtered spectrum
        plt.subplot(2, 3, 4)
        plt.imshow(np.log1p(np.abs(filtered_f_shift)), cmap='viridis')
        plt.title('Filtered Spectrum (log scale)')
        
        # Filtered image
        plt.subplot(2, 3, 5)
        plt.imshow(img_filtered, cmap='gray')
        plt.title('Filtered Image')
        
        plt.tight_layout()
        plt.show()
    
    return img_filtered

if __name__ == "__main__":
    # Path to your image
    image_path = "Grayscale_8bits_palette_sample_image.png"  # Replace with your image path
    
    # Define notch centers in the frequency domain
    # These are example coordinates that you should adjust based on your specific image
    # Each (u, v) defines where a notch filter should be applied
    notch_centers = [
        (50, 50),   # Example notch 1
        (-50, -50),  # Example notch 2 (symmetric to the first one)
        (100, 20),   # Example notch 3
        (-100, -20)  # Example notch 4 (symmetric to the third one)
    ]
    
    # Apply the notch filter
    filtered_image = apply_notch_filter(image_path, notch_centers, d0=10)
    
    # Save the filtered image
    cv2.imwrite("notch_filtered_image.jpg", filtered_image)
    print("Filtered image saved as 'notch_filtered_image.jpg'")