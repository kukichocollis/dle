import matplotlib.pyplot as plt 
from matplotlib.image import imread

# sample image
image = imread('PetImages/train/dog/1.jpg')
image.shape
plt.imshow(image)
plt.show()

from tensorflow.keras import models, layers

train_dir = './Petimages/train'
test_dir = './Petimages/test'

model = models.Sequential()
model.add(layers.Conv2D(32,(3,3), activation='relu', input_shape=(128,128,3)))
model.add(layers.MaxPooling2D(2,2))
model.add(layers.Conv2D(64,(3,3), activation='relu'))
model.add(layers.MaxPooling2D(2,2))
model.add(layers.Flatten())
model.add(layers.Dense(units=512, activation='relu'))
model.add(layers.Dense(units=1, activation='sigmoid'))

model.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
model.summary()

#### Data augmentation
# https://machinelearningmastery.com/image-augmentation-deep-learning-keras/
#
from tensorflow.keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(rescale = 1./255, 
                                   shear_range = 0.2,
                                   zoom_range = 0.2, 
                                   horizontal_flip = True)

test_datagen = ImageDataGenerator(rescale = 1./255)

train_generator = train_datagen.flow_from_directory(
    train_dir,                      
    target_size=(128, 128), # (180,180)
    batch_size=20,          # 32,...
    class_mode = 'binary')

test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=(128, 128),
    batch_size=20,
    class_mode = 'binary')

#
# Visualize data from data generator
# 1. Extract one batch
for x_data, t_data in train_generator:
    print(x_data.shape)  # (20, 128, 128, 3)
    print(type(x_data))  # <class 'numpy.ndarray'>
    print(t_data)        # [0. 1. 1. 1. 1. 0. 0. 1. 0. 0. 1. 0. 0. 1. 1. 0. 0. 0. 1. 0.]
    # 0 : 고양이,  1 : 댕댕이
    # break

# 2. Display images in the batch

fig = plt.figure(figsize=(15, 12))
axs = []

for x_data, t_data in train_generator:
    for idx, img in enumerate(x_data):
        axs.append(fig.add_subplot(4,5,idx+1))
        plt.imshow(img)
    break
plt.show()

#
######################################################
# Training model using augmentated data
######################################################
#
history = model.fit(
    train_generator, 
    # steps_per_epoch = 100,  # len(x_data)/batch_size = 2000/20 = 100
    epochs=10, # 100
    validation_data=test_generator, 
    validation_steps=5)

#
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.xlabel('Epoch')
plt.xlabel('Accuracy')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()

