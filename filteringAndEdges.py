import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt


def change_contrast(image, alpha, beta):
    # Create a new image to return
    output_image = np.zeros(image.shape, dtype=np.uint8)

    # Traverse the entire image
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            # Adjust the contrast and brightness
            output_image[i, j] = np.clip(alpha * image[i, j] + beta, 0, 255)

    # Return the image
    return output_image


def my_roberts(image):
    # Set the filters
    roberts_kernel_x = np.array([[1, 0], [0, -1]], dtype=np.float32)
    roberts_kernel_y = np.array([[0, 1], [-1, 0]], dtype=np.float32)

    # Get the dimensions of the image
    height, width = gray_img.shape

    # Create an empty image to return
    img_roberts_edges = np.zeros((height, width), dtype=np.uint8)

    # Traverse all points and perform convolution with Roberts filters
    for y in range(1, height - 1):
        for x in range(1, width - 1):
            gx = (roberts_kernel_x[0, 0] * gray_img[y - 1, x - 1]) + (roberts_kernel_x[0, 1] * gray_img[y - 1, x]) + \
                 (roberts_kernel_x[1, 0] * gray_img[y, x - 1]) + (roberts_kernel_x[1, 1] * gray_img[y, x])
            gy = (roberts_kernel_y[0, 0] * gray_img[y - 1, x - 1]) + (roberts_kernel_y[0, 1] * gray_img[y - 1, x]) + \
                 (roberts_kernel_y[1, 0] * gray_img[y, x - 1]) + (roberts_kernel_y[1, 1] * gray_img[y, x])
            mag = np.sqrt(gx ** 2 + gy ** 2)
            img_roberts_edges[y, x] = mag

    # Normalize the magnitude image [0, 255]
    img_roberts_edges = np.uint8(img_roberts_edges * 255.0 / np.max(img_roberts_edges))

    # Amplitude thresholding to obtain edges
    threshold = 50
    img_roberts_edges[img_roberts_edges < threshold] = 0
    img_roberts_edges[img_roberts_edges >= threshold] = 255

    # Return the image
    return img_roberts_edges


def my_prewitt(image):
    # Define the filters
    h_kernel = np.array([[-1, 0, 1],
                         [-1, 0, 1],
                         [-1, 0, 1]], dtype=np.float32)
    v_kernel = np.array([[-1, -1, -1],
                         [0, 0, 0],
                         [1, 1, 1]], dtype=np.float32)

    # Get the dimensions of the image
    height, width = image.shape[:2]

    # Create a new empty image
    edges = np.zeros((height, width), dtype=np.float32)

    # Traverse all points in the image
    for y in range(1, height - 1):
        for x in range(1, width - 1):
            # Apply horizontal and vertical filters centered at (x, y)
            h_value = np.sum(image[y - 1:y + 2, x - 1:x + 2] * h_kernel)
            v_value = np.sum(image[y - 1:y + 2, x - 1:x + 2] * v_kernel)

            # Calculate the magnitude
            magnitude = np.sqrt(h_value ** 2 + v_value ** 2)

            # Store the magnitude
            edges[y, x] = magnitude

    # Normalize the edges
    edges = np.uint8(edges * 255.0 / np.max(edges))

    # Amplitude thresholding
    threshold = 50
    edges[edges > threshold] = 255
    edges[edges <= threshold] = 0

    # Return the image
    return edges


def my_sobel(image):
    # Sobel filters
    kernel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    kernel_y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])

    # Add zero-padding around the image
    padded_image = np.pad(image, (1, 1), mode='constant')

    # Create an image to return
    output_image = np.zeros(image.shape)

    # Traverse all points and apply the filter
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            gx = np.sum(kernel_x * padded_image[i:i + 3, j:j + 3])
            gy = np.sum(kernel_y * padded_image[i:i + 3, j:j + 3])
            output_image[i, j] = np.sqrt(gx ** 2 + gy ** 2)

    # Normalize the image
    output_image = output_image / np.max(output_image)

    return output_image


def canny(image, lower_threshold, upper_threshold):
    edges = cv.Canny(image, lower_threshold, upper_threshold)
    return edges


# Load the image
img = cv.imread('lenna.png')
gray_img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

# Contrast and brightness values
alphaIN = 1
betaIN = 1

gray_img = change_contrast(gray_img, alphaIN, betaIN)

# Apply filters/detectors
img_roberts = my_roberts(gray_img)

img_prewitt = my_prewitt(gray_img)

img_sobel = my_sobel(gray_img)

img_canny1 = canny(gray_img, 100, 200)
img_canny2 = canny(gray_img, 50, 150)
img_canny3 = canny(gray_img, 150, 250)
img_canny4 = canny(gray_img, 50, 250)


# Original
img_with_edges = cv.cvtColor(gray_img, cv.COLOR_GRAY2BGR)

plt.subplot(2, 4, 1), plt.imshow(img_with_edges, cmap='gray')
plt.title('Original'), plt.xticks([]), plt.yticks([])


# Roberts
img_with_edges = cv.cvtColor(gray_img, cv.COLOR_GRAY2BGR)

img_with_edges[np.where(img_roberts > 0)] = [255, 0, 0]

plt.subplot(2, 4, 2), plt.imshow(img_with_edges, cmap='gray')
plt.title('Roberts Edges'), plt.xticks([]), plt.yticks([])


# Prewitt
img_with_edges = cv.cvtColor(gray_img, cv.COLOR_GRAY2BGR)

img_with_edges[np.where(img_prewitt > 0)] = [255, 0, 0]

plt.subplot(2, 4, 3), plt.imshow(img_with_edges, cmap='gray')
plt.title('Prewitt Edges'), plt.xticks([]), plt.yticks([])


# Sobel
img_with_edges = cv.cvtColor(gray_img, cv.COLOR_GRAY2BGR)

img_with_edges[np.where(img_sobel > 0)] = [255, 0, 0]

plt.subplot(2, 4, 4), plt.imshow(img_with_edges, cmap='gray')
plt.title('Sobel Edges'), plt.xticks([]), plt.yticks([])


# Canny (100, 200)
plt.subplot(2, 4, 5), plt.imshow(img_canny1, cmap='gray')
plt.title('Canny (100, 200)'), plt.xticks([]), plt.yticks([])


# Canny (50, 150)
plt.subplot(2, 4, 6), plt.imshow(img_canny2, cmap='gray')
plt.title('Canny (50, 150)'), plt.xticks([]), plt.yticks([])


# Canny (150, 250)
plt.subplot(2, 4, 7), plt.imshow(img_canny3, cmap='gray')
plt.title('Canny (150, 250)'), plt.xticks([]), plt.yticks([])


# Canny (50, 250)
plt.subplot(2, 4, 8), plt.imshow(img_canny4, cmap='gray')
plt.title('Canny (50, 250)'), plt.xticks([]), plt.yticks([])


plt.show()
