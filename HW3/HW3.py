import cv2
import numpy as np
import matplotlib.pyplot as plt


def is_grayscale(img):
    if len(img.shape) == 2:
        return True
    elif len(img.shape) == 3:
        if np.all(img[:, :, 0] == img[:, :, 1]) and np.all(img[:, :, 0] == img[:, :, 2]):
            return True
    return False


file_name = input('Please enter the name of an image file : ')
C = cv2.imread(file_name)

if is_grayscale(C):
    C = cv2.cvtColor(C, cv2.COLOR_BGR2GRAY)
    C_prime = cv2.equalizeHist(C)
else:
    G = cv2.cvtColor(C, cv2.COLOR_BGR2GRAY)
    G_prime = cv2.equalizeHist(G)
    C_prime = C.astype(np.float64)

    G[G == 0] = 1
    ratio = G_prime / G

    np.set_printoptions(threshold=np.inf)

    for i in range(3):
        C_prime[:, :, i] = C_prime[:, :, i] * ratio

    C_prime[C_prime > 255] = 255
    C_prime[C_prime < 0] = 0
    C_prime = np.round(C_prime).astype(np.uint8)


cv2.imshow('Original Image', C)
C_hist, C_bins = np.histogram(C, bins=256)
plt.subplot(1, 2, 1)
plt.title('Original')
plt.bar(C_bins[:-1], C_hist, width=np.diff(C_bins))

cv2.imshow('Equalized Image', C_prime)
C_prime_hist, C_prime_bins = np.histogram(C_prime, bins=256)
plt.subplot(1, 2, 2)
plt.title('Equalized')
plt.bar(C_prime_bins[:-1], C_prime_hist, width=np.diff(C_prime_bins))
plt.tight_layout()
plt.show()
cv2.waitKey(0)
cv2.destroyAllWindows()
