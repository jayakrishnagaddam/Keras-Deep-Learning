import keras
from keras.models import mnist
from keras import models
from keras import layers

(train_images,train_labels), (test_images, test_labels)=mnist.load_data()


model=models.Sequential()
model.add(layers.Dense(510, activation='relu', input_shape=28*28 ))
model.add(layers.Dense(10, activation='softmax'))



model.compile(
    optimizer='rmsprop',
    loss='categorical_crossentropy',
    metrics=['accuracy']
    )


model.fit(train_images, train_labels, epochs=10, batch_size=5)

