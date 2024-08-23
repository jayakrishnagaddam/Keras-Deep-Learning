import keras
from keras.datasets import imdb
from keras import models
import numpy as np
from keras import layers

# Load the dataset with a limit of the top 2,000 most frequent words
(train_data, train_labels), (test_data, test_labels) = imdb.load_data(num_words=15000)

# Set the dimensions to 2000
dimensions = 15000

def preprocess(sequences, dimensions):
    results = np.zeros((len(sequences), dimensions))
    for i, sequence in enumerate(sequences):
        results[i, sequence] = 1
    return results

x_train = preprocess(train_data, dimensions)
x_test = preprocess(test_data, dimensions)

y_train = np.asarray(train_labels).astype('float32')
y_test = np.asarray(test_labels).astype('float32')

model = models.Sequential()

model.add(layers.Dense(16, activation='relu', input_shape=(dimensions,)))
model.add(layers.Dense(16, activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))

model.compile(
    optimizer='rmsprop',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

model.fit(x_train, y_train, epochs=4, batch_size=512)

results = model.evaluate(x_test, y_test)

new_review = input("Enter your review: ")
word_index = imdb.get_word_index()

def review_to_sequence(review, word_index, max_words=dimensions):
    words = review.lower().split()
    sequence = [word_index.get(word, 0) for word in words]
    sequence = [index for index in sequence if index < max_words]
    return sequence

new_review_sequence = review_to_sequence(new_review, word_index)

def preprocess_new_review(sequence, dimensions):
    results = np.zeros((1, dimensions))  
    results[0, sequence] = 1  
    return results

x_new_review = preprocess_new_review(new_review_sequence, dimensions)
prediction = model.predict(x_new_review)
print(prediction)

if prediction >= 0.5:
    print("Positive review")
else:
    print("Negative review")
