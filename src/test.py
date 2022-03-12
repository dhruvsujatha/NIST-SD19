import numpy as np
from PIL import Image as img
from PIL import ImageOps as imgops
import math

pixels = 50

test_image = img.open("C:/NN/NIST SD 19/Words/Test 0.jpg")
test_image = imgops.grayscale(test_image)
test_image = np.asarray(test_image)

print(np.shape(test_image))

test_image = np.asarray((img.fromarray(test_image)).resize((math.ceil(pixels * (np.shape(test_image)[1] / np.shape(test_image)[0])), pixels)))
print(np.shape(test_image))
print(np.shape(test_image[:, 0:30]))

for i in range(0, math.ceil(pixels * (np.shape(test_image)[1] / np.shape(test_image)[0])) - pixels, 3):
    print(i)
    img.fromarray(test_image[:, i:(i + pixels)]).show()
    print(np.shape(test_image[:, i:(i + pixels)]))
