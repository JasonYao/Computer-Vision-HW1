# HW1 for Computer Vision

import cv2
import numpy as np
from matplotlib import pyplot as plt
import math

# Asserts to test that the values are correct
# assert np.abs(np.sum(psi)) < 1e-8
# assert np.abs(np.sum(np.abs(psi) ** 2) - 1) < 1e-8

# Tutorial stuff to get used to cv2's API

# Loads an image
# grayscale image
# img = cv2.imread('small.jpg', cv2.IMREAD_COLOR)

# Displays an image
# cv2.imshow('image', img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
# plt.imshow(img, cmap = 'gray', interpolation = 'bicubic')
# plt.xticks([]), plt.yticks([])  # to hide tick values on X and Y axis
# plt.show()

# cv2.imshow("sigma: " + str(sigma) + " - theta: " + str(theta), out_img)
# plt.imshow(np.real(main_kernel), cmap=plt.get_cmap('gray'))
# plt.show()

WINDOW_SIZE = 37
HALF_SIZE = WINDOW_SIZE // 2
SIZE = 168
INPUT_IMAGE = "img/noisy_circle.jpg"
PENTAGON_LEFT = "img/left.png"
PENTAGON_RIGHT = "img/right.png"
EPSILON = -5000
NORMALISATION = 4000


def calculate_b(u_squared, sigma):
    return np.exp(-1 * u_squared/(2 * (np.power(sigma, 2))))


def calculate_a(u, sigma):
    return np.exp(1j * (np.pi/(2 * sigma)) * u)


def q1_make_wavelet(sigma, theta):
    x_range = list(range(-HALF_SIZE, HALF_SIZE + 1))
    y_range = list(range(-HALF_SIZE, HALF_SIZE + 1))

    # Generates the matrix
    [x, y] = np.meshgrid(x_range, y_range)

    # Generates intermediary values
    e_theta = (np.cos(theta), np.sin(theta))
    u = x * e_theta[0] + y * e_theta[1]
    u_squared = np.power(x, 2) + np.power(y, 2)

    # Let a = e^(i * (pi/(2*sigma)) * ue)
    a = calculate_a(u, sigma)

    # Let b = e^(-(u^2/(2*(sigma^2))))
    b = calculate_b(u_squared, sigma)

    c2 = np.sum(a * b) / np.sum(b)
    # TODO double check c1
    c1 = 1 / np.sqrt(np.sum((1 - 2 * c2 * np.cos(u * np.pi / (2 * sigma)) + np.power(c2, 2)) * np.exp(-1 * u_squared / (1 * np.power(sigma, 2)))))

    return np.transpose((c1 / sigma) * (a - c2) * b)


def init_kernel(sigma, theta):
    kernel = np.zeros([WINDOW_SIZE, WINDOW_SIZE], np.complex)
    x_range = range(-HALF_SIZE, HALF_SIZE + 1)
    y_range = range(-HALF_SIZE, HALF_SIZE + 1)

    numerator = 0
    denominator = 0

    for x in x_range:
        for y in y_range:
            b = calculate_b(np.power(x, 2) + np.power(y, 2), sigma)
            # TODO double check numerator
            numerator += (np.cos((np.pi/(2*sigma))*np.dot([x, y], [np.cos(theta), np.sin(theta)])) + (1j*np.sin((np.pi/(2*sigma))*np.dot([x, y], [np.cos(theta), np.sin(theta)]))))*b
            denominator += b

    c2 = numerator / denominator

    psi = 0
    for x in x_range:
        for y in y_range:
            psi += (1-(2*c2*np.cos((np.pi/(2*sigma))*np.dot([x, y], [np.cos(theta), np.sin(theta)])))+(c2**2))*np.exp(-(((x**2)+(y**2))/(sigma**2)))

    c1 = 1 / np.sqrt(psi)

    for x in x_range:
        for y in y_range:
            kernel[x + HALF_SIZE][y + HALF_SIZE] = (c1 / sigma) * (np.cos((np.pi / (2 * sigma)) * np.dot([x, y], [np.cos(theta), np.sin(theta)])) + (1j * np.sin((np.pi / (2 * sigma)) * np.dot([x, y], [np.cos(theta), np.sin(theta)]))) - c2) * calculate_b(np.power(x, 2) + np.power(y, 2), sigma)
    return kernel


def print_q2_image(img, sigma, theta, count, hist_real, hist_imag, should_print_image, size):
    # Output image
    out_img = np.zeros(img.shape[:2], np.float64)  # Matrix of black pixels the same size as the input image

    main_kernel = init_kernel(sigma, theta)
    q1_kernel = q1_make_wavelet(sigma, theta)
    # We now normalise with 0 and output the real image
    normalisation = cv2.normalize(main_kernel.real, np.zeros((WINDOW_SIZE, WINDOW_SIZE)), alpha=0, beta=255,
                                  norm_type=cv2.NORM_MINMAX)
    out_img = cv2.filter2D(img, -1, np.real(main_kernel), out_img, (-1, -1), cv2.BORDER_DEFAULT)

    if should_print_image:
        out_img = add_text_to_img(out_img, "S: " + str(sigma) + ", T: " + str("%.2f" % theta) + ", real")
        cv2.imwrite("output/out_img_real_" + str(count) + ".jpg", out_img)
        plt.imshow(np.real(main_kernel), cmap=plt.get_cmap('gray'))
        plt.title("S: " + str(sigma) + ", T: " + str("%.2f" % theta) + ", real")
        plt.savefig("output/out_kernel_real_" + str(count) + ".png")
        plt.gcf().clear()

    hist_real = print_q3_histogram(out_img, hist_real, size)

    # We now normalise with 1 and output the imaginary image
    out_img = cv2.filter2D(img, -1, np.imag(main_kernel), out_img, (-1, -1), cv2.BORDER_DEFAULT)

    if should_print_image:
        out_img = add_text_to_img(out_img, "S: " + str(sigma) + ", T: " + str("%.2f" % theta) + ", imag")
        cv2.imwrite("output/out_img_imag_" + str(count) + ".jpg", out_img)
        plt.imshow(np.imag(main_kernel), cmap=plt.get_cmap('gray'))
        plt.title("S: " + str(sigma) + ", T: " + str("%.2f" % theta) + ", imag")
        plt.savefig("output/out_kernel_imag_" + str(count) + ".png")
        plt.gcf().clear()

    hist_imag = print_q3_histogram(out_img, hist_imag, size)

    return np.dstack((hist_real, hist_imag)), hist_real, hist_imag


def gaussian_blur(img):
    out = np.zeros(img.shape[:2], np.float64)
    out = cv2.GaussianBlur(img, ((2 * HALF_SIZE) + 1, (2 * HALF_SIZE) + 1), 0, out)
    return out


def print_q3_histogram(img, hist, size):
    for x in range(size):
        for y in range(size):
            hist[x][y] = max(hist[x][y], img[x][y])
    return hist


def detect_edges(real, imaginary, size):
    epsilon = EPSILON

    ratio = np.zeros((size, size))
    edge_array = np.zeros((size, size))

    for i in range(size):
        for j in range(size):
            ratio[i][j] = (real[i][j] + (0.001 + epsilon)) / (imaginary[i][j] + epsilon)
            diff = np.amax(ratio)
            edge_array[i][j] = 1 - (ratio[i][j] / diff)

    return edge_array * NORMALISATION


def get_values(img, should_print_images):
    # Image dimensions
    height, width = img.shape

    # 12 combinations of parameters, let lambda_{0 -> 11} = (sigma, theta)
    sigmas = [1, 3, 6]
    thetas = [0, math.pi / 4, math.pi / 2, (3 * math.pi) / 4]

    count = 0

    hist_real = np.ones(img.shape[:2], np.float64)  # array to hold min of weights for each real pixel
    hist_imag = np.zeros(img.shape[:2], np.float64)  # array to hold max of weights for each imaginary pixel
    hist = np.zeros(img.shape[:2], np.float64)

    for sigma in sigmas:
        for theta in thetas:
            hist, hist_real, hist_imag = print_q2_image(img, sigma, theta, count, hist_real, hist_imag, should_print_images, height - 1)
            count += 1

    return hist, hist_real, hist_imag, height, width


def add_text_to_img(img, text):
    height, width = img.shape

    if isinstance(text, list):
        # Multiple text line
        number_of_text_lines = len(text)
        text = text[::-1]

        for text_line_index in range(number_of_text_lines):
            cv2.putText(img, text[text_line_index], (0, height - (10 * (text_line_index + 1))),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_4)
    else:
        cv2.putText(img, text, (0, height - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_4)

    return img


def main():
    # Q1: Now we create all the different morlet wavelets
    print("Q1: Generating Morlet Wavelet images now")
    circle_img = cv2.imread(INPUT_IMAGE, cv2.IMREAD_GRAYSCALE)
    circle_hist, circle_hist_real, circle_hist_imag, circle_height, circle_width = get_values(circle_img, True)

    # Q2: Now we output the original image with a gaussian blur
    print("Q2: Generating gaussian-blurred image now")
    gaussian_img = gaussian_blur(circle_img)
    gaussian_img = add_text_to_img(gaussian_img, "Gaussian Blur")
    cv2.imwrite("output/gaussian.jpg", gaussian_img)

    # Q3: Now we print out histograms for real and imaginary values
    print("Q3: Generating real & imaginary value histograms now")
    print("\tPlease wait, outputting real histogram")
    plt.hist(circle_hist[:, :, 0], 10)
    plt.title("Real Values Histogram")
    plt.xlabel("Value")
    plt.ylabel("Frequency")
    plt.savefig("output/histogram_real.png")
    plt.gcf().clear()

    print("\tPlease wait, outputting imaginary histogram")
    plt.hist(circle_hist[:, :, 1], 10)
    plt.title("Imaginary Values Histogram")
    plt.xlabel("Value")
    plt.ylabel("Frequency")
    plt.savefig("output/histogram_imaginary.png")
    plt.gcf().clear()

    # Q4a: Edge detection for input image
    print("Q4a: Generating edge-detection for the input image")
    edge_detection_circle = detect_edges(circle_hist_real, circle_hist_imag, circle_height - 1)
    edge_detection = add_text_to_img(edge_detection_circle, ["Edge Detection:", "Circle"])
    cv2.imwrite("output/edge.jpg", edge_detection)

    # Q4b: Edge detection for pentagon image
    print("Q4b: Generating edge-detection for the pentagon (takes ~20 seconds)")
    pentagon_img = cv2.imread(PENTAGON_LEFT, cv2.IMREAD_GRAYSCALE)
    pentagon_hist, pentagon_hist_real, pentagon_hist_imag, pentagon_height, pentagon_width = get_values(pentagon_img, False)
    pentagon_edge_detection = detect_edges(pentagon_hist_real, pentagon_hist_imag, pentagon_height)

    pentagon_edge_detection = add_text_to_img(pentagon_edge_detection, ["Edge Detection:", "Pentagon"])
    cv2.imwrite("output/edge_pentagon.jpg", pentagon_edge_detection)


if __name__ == '__main__':
    main()
