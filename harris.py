import cv2
import numpy as np
import Frequency

def harris_corner_detector(image_path, block_size=2, k_size=3, k=0.04, threshold=0.01):
    gray_image= cv2.imread(image_path)
    # Convert to grayscale
    gray_image = Frequency.prepare(image_path)

    # Compute the x and y derivatives
    gray_image = np.float32(gray_image)
    Ix = cv2.Sobel(gray_image, cv2.CV_64F, 1, 0, ksize=k_size)
    Iy = cv2.Sobel(gray_image, cv2.CV_64F, 0, 1, ksize=k_size)

    # Compute the products of derivatives at every pixel
    Ix2 = Ix * Ix
    Iy2 = Iy * Iy
    Ixy = Ix * Iy

    # Compute the sum of the products of derivatives for each neighborhood
    Sx2 = cv2.boxFilter(Ix2, cv2.CV_64F, (block_size, block_size))
    Sy2 = cv2.boxFilter(Iy2, cv2.CV_64F, (block_size, block_size))
    Sxy = cv2.boxFilter(Ixy, cv2.CV_64F, (block_size, block_size))

    # Compute the response function R
    det = Sx2 * Sy2 - Sxy * Sxy
    trace = Sx2 + Sy2
    R = det - k * trace * trace

    # Threshold and get the coordinates of corner points
    corner_threshold = threshold * R.max()
    corner_points = np.where(R > corner_threshold)
    corner_points = np.stack((corner_points[1], corner_points[0]), axis=-1)

    for point in corner_points:
        x, y = point
        cv2.circle(gray_image, (x, y), 2, (0, 255, 0), thickness=-1)
    cv2.imwrite('images/image_harris.png',gray_image)

    return 'images/image_harris.png'



