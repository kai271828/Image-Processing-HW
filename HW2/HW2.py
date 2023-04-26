import numpy as np
import cv2

file_name = input('Please enter the name of an image file : ')
I = cv2.imread(file_name)
I = cv2.cvtColor(I, cv2.COLOR_BGR2GRAY)
I_prime = I.copy()

D2 = np.array([
    [0, 128, 32, 160],
    [192, 64, 224, 96],
    [48, 176, 16, 144],
    [240, 112, 208, 80]
])
D = np.tile(D2, ((I.shape[0] // D2.shape[0]) + 1, (I.shape[1] // D2.shape[1]) + 1))
D = D[:I.shape[0], :I.shape[1]]

I_prime[I_prime > D] = 255
I_prime[I_prime <= D] = 0

'''
resized = cv2.resize(I_prime, (128, 128), interpolation=cv2.INTER_AREA)

with open('output.txt', 'w') as file:
    for i in range(resized.shape[0]):
        for j in range(resized.shape[1]):
            if resized[i, j] >= 128:
                print(' ', end='', file=file)
            else:
                print('.', end='', file=file)
        print('', file=file)
'''

cv2.imshow('I', I)
cv2.imshow("I'", I_prime)
cv2.waitKey(0)
cv2.destroyAllWindows()

Q = I // 85

D1 = np.array([
    [0, 56],
    [84, 28]
])
D = np.tile(D1, ((I.shape[0] // D1.shape[0]) + 1, (I.shape[1] // D1.shape[1]) + 1))
D = D[:I.shape[0], :I.shape[1]]

I_prime[I_prime - 85 * Q > D] = 1
I_prime[I_prime - 85 * Q <= D] = 0
I_prime += Q

I_prime *= 85

cv2.imshow("I'", I_prime)
cv2.waitKey(0)
cv2.destroyAllWindows()
