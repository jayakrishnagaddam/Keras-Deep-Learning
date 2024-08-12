from keras.datasets import mnist
from keras import layers
from keras import models

(train_images,train_images),(test_images, test_labels)=mnist.load_data()


model=models.Sequential()
