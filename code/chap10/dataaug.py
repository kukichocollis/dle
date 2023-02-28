#
# https://machinelearningmastery.com/image-augmentation-deep-learning-keras/
#
import matplotlib.pyplot as plt
from numpy import expand_dims
from tensorflow.keras.preprocessing.image import load_img, img_to_array

image = load_img("dog.jpg")
array = img_to_array(image)
array.shape
sample = expand_dims(array, 0)  # batch image
sample.shape

from tensorflow.keras.preprocessing.image import ImageDataGenerator

datagen = ImageDataGenerator(rescale = 1./255,
    rotation_range=90, brightness_range=[0.8, 1.0],
    width_shift_range=0.2, zoom_range=[0.8, 1.2],
    height_shift_range=0.2)

obj = datagen.flow(sample, batch_size=1)

fig = plt.figure(figsize=(20,5))
for i in range(8):
    plt.subplot(1,8,i+1)
    image = obj.next()
    # print(image.shape)
    plt.imshow(image[0])
