from keras.preprocessing.image import ImageDataGenerator
from keras import models, layers

train_datagen = ImageDataGenerator(rescale=1./255)
validation_datagen = ImageDataGenerator(rescale=1./255)
test_datagen = ImageDataGenerator(rescale=1./255)


train_generator = train_datagen.flow_from_directory(
    'dataset\Train',
    target_size=(28, 28),
    batch_size=20,
    class_mode='binary'
)

validation_generator = validation_datagen.flow_from_directory(
    'dataset/Validation',
    target_size=(28, 28),
    batch_size=20,
    class_mode='binary'
)

test_generator = test_datagen.flow_from_directory(
    'dataset/Test',
    target_size=(28, 28),
    batch_size=20,
    class_mode='binary'
)



model = models.Sequential()
model.add(layers.Flatten(input_shape=(28, 28, 3))) 
model.add(layers.Dense(510, activation='relu'))
model.add(layers.Dense(1, activation='sigmoid')) 

model.compile(
    optimizer='rmsprop',
    loss='binary_crossentropy',
    metrics=['accuracy']
)



history = model.fit(
    train_generator,
    steps_per_epoch=100,
    epochs=10,
    validation_data=validation_generator,
    validation_steps=50
)



test_loss, test_acc = model.evaluate(test_generator, steps=50)
print('Test accuracy:', test_acc)




