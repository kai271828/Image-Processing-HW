import cv2
import numpy as np


def unsharp_masking(original, mode, u_kernel_size, u_k, verbose=False):
    if mode == 'average':
        unsharp_mask = np.ones((u_kernel_size, u_kernel_size), np.float32) * -u_k / u_kernel_size ** 2
        mid = u_kernel_size // 2
        unsharp_mask[mid, mid] = 1 + (u_kernel_size ** 2 - 1) * u_k / u_kernel_size ** 2
        if verbose:
            print(1 + (u_kernel_size ** 2 - 1) * u_k / u_kernel_size ** 2)
            print(f'unsharp_mask: {unsharp_mask}')

        result = cv2.filter2D(original, -1, unsharp_mask)
        return result

    elif mode == 'median':
        blurred = cv2.medianBlur(original, u_kernel_size)

        if verbose:
            cv2.imshow(f'{u_kernel_size} * {u_kernel_size} Blurred', blurred)

        g = (original.astype(np.float32) - blurred.astype(np.float32)) * u_k

        if verbose:
            print(g)

        result = original.astype(np.float32) + g
        result[result > 255] = 255.0
        result[result < 0] = 0

        return result.astype(np.uint8)

    else:
        print('mode error')
        raise RuntimeError


file_name = input('Please enter the name of an image file : ')
image = cv2.imread(file_name)

kernel_size = 7

avg_filter = np.ones((kernel_size, kernel_size), np.float32) / kernel_size ** 2
a = cv2.filter2D(image, -1, avg_filter)

m = cv2.medianBlur(image, kernel_size)

cv2.imshow(f'{kernel_size} * {kernel_size} Average Filter', a)
cv2.imshow(f'{kernel_size} * {kernel_size} Median Filter', m)

u = unsharp_masking(image, 'median', kernel_size, 0.8)

cv2.imshow(f'{kernel_size} * {kernel_size} Unsharp Masking', u)
cv2.waitKey(0)
cv2.destroyAllWindows()

for m in ['average']:
    for ks in range(7, 8, 1):
        for k in [2, 10, 20, 30]:
            u = unsharp_masking(image, m, ks, k / 10)
            cv2.imshow(f'{ks} * {ks} {m} Filter, k = {k / 10}', u)

cv2.waitKey(0)
cv2.destroyAllWindows()
