import numpy as np
import matplotlib.pyplot as plt

# 1. Input a color image C(R,G,B)
file_name = input('Please enter the name of an image file : ')
C = plt.imread(file_name)

# 2. Output the color image C
plt.imshow(C)
plt.axis('off')
plt.show()


# 3. Transform the color image C into a grayscale image I by I = (R+G+B)/3
def rgb2grayscale(image):
    I = np.sum(image, axis=2) / 3
    return I


I = rgb2grayscale(C)

# 4. Show the grayscale image I
plt.imshow(I, cmap='gray')
plt.axis('off')
plt.show()
