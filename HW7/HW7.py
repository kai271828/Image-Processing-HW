import cv2
import numpy as np
import matplotlib.pyplot as plt


def zero_mean_gaussian_noise(img, sigma):

    def transform(value):
        if value < 0:
            return 0
        elif value > 255:
            return 255
        else:
            return np.round(value)

    noise = img.copy()
    for i in range(img.shape[0]):
        for j in range(0, img.shape[1], 2):
            r = np.random.random()
            phi = np.random.random()
            z1 = sigma * np.cos(2 * np.pi * phi) * np.sqrt(-2 * np.log(r))
            z2 = sigma * np.sin(2 * np.pi * phi) * np.sqrt(-2 * np.log(r))
            noise[i, j] = transform(img[i, j] + z1)
            noise[i, j + 1] = transform(img[i, j] + z2)

    return noise


def draw_histogram(img, title):
    hist, bins = np.histogram(img, bins=256)
    plt.title(title)
    plt.xlim(0, 255)
    plt.bar(bins[:-1], hist, width=1)
    plt.show()


img_size = (300, 300)
g = np.zeros(img_size, np.uint8)
g += 100
f = zero_mean_gaussian_noise(g, 200)
cv2.imshow('Image g', g)
cv2.imshow('Noisy Image f', f)
cv2.waitKey(0)
cv2.destroyAllWindows()

draw_histogram(g, 'Image g')
draw_histogram(f, 'Noisy Image f')
