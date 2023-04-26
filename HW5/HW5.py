import cv2
import numpy as np

img_size = (500, 500)
img_center = (img_size[0] / 2, img_size[1] / 2)

# create a black background image
img = np.zeros(img_size, np.uint8)

img = cv2.imread('duck.png')
img = cv2.resize(img, img_size)
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# create a white square in the center of the image
# cv2.rectangle(img, (100, 100), (400, 400), (255, 255, 255), -1)

# rotate function
def rotate(image, size, degree, center, interpolation):

    rotated_image = np.zeros(size, np.uint8)
    radian = np.deg2rad(degree)

    def tl2cen(cor):
        return cor[0] - center[0], cor[1] - center[1]

    def cen2tl(cor):
        return cor[0] + center[0], cor[1] + center[1]

    def transform(cor, r):
        cor = tl2cen(cor)
        cor = np.cos(r) * cor[0] + np.sin(r) * cor[1], -np.sin(r) * cor[0] + np.cos(r) * cor[1]
        return cen2tl(cor)

    def bilinear_interpolation(image, xf, yf):
        x_left = int(np.floor(xf))
        x_right = int(np.ceil(xf))
        y_top = int(np.floor(yf))
        y_bottom = int(np.ceil(yf))

        if 0 <= x_left < size[0] and 0 <= x_right < size[0] and 0 <= y_top < size[1] and 0 <= y_bottom < size[1]:
            alpha = xf - x_left
            v_top = alpha * image[x_right][y_top] + (1 - alpha) * image[x_left][y_top]
            v_bottom = alpha * image[x_right][y_bottom] + (1 - alpha) * image[x_left][y_bottom]
            beta = yf - y_top
            return beta * v_bottom + (1 - beta) * v_top
        else:
            return 0

    original_coordinate = [[transform((i, j), radian) for j in range(size[1])] for i in range(size[0])]
    for i in range(size[0]):
        for j in range(size[1]):
            x, y = original_coordinate[i][j]

            if interpolation == 'neighbor':
                x, y = int(np.round(x)), int(np.round(y))
                if 0 <= x < size[0] and 0 <= y < size[1]:
                    rotated_image[i, j] = image[x, y]
            elif interpolation == 'bilinear':
                rotated_image[i, j] = bilinear_interpolation(image, x, y)
            else:
                print('Interpolation Error.')
                return
    return rotated_image


rotated_nn = rotate(img, img_size, 30, img_center, 'neighbor')
rotated_bl = rotate(img, img_size, 30, img_center, 'bilinear')

cv2.imshow('Original', img)
cv2.imshow('Rotation with neighbor interpolation', rotated_nn)
cv2.imshow('Rotation with bilinear interpolation', rotated_bl)

# define the rotation matrix and rotate the image by 30 degrees with neighbor interpolation
M = cv2.getRotationMatrix2D(img_center, 30, 1)
cv_rotated_nn = cv2.warpAffine(img, M, img_size, flags=cv2.INTER_NEAREST)
cv_rotated_bl = cv2.warpAffine(img, M, img_size, flags=cv2.INTER_LINEAR)

# display the rotated images made by OpenCV
cv2.imshow('Rotation with neighbor interpolation by OpenCV', cv_rotated_nn)
cv2.imshow('Rotation with bilinear interpolation by OpenCV', cv_rotated_bl)
cv2.waitKey(0)
