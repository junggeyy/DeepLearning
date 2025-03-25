import tensorflow as tf
from tensorflow import keras
from keras import layers

# Loading MNIST data from tensorflow.keras.datasets
# training set                 # test set
(train_images, train_labels), (test_images, test_labels) = keras.datasets.mnist.load_data()

# preapring the image (input) data
train_images = train_images.reshape((60000, 28 * 28)) # reshape the structure of data from 2D matrix to 1D vector
train_images = train_images.astype("float32") / 255    # change the data type from int to float32 and normalize the value to [0,1]
test_images = test_images.reshape((10000, 28 * 28))
test_images = test_images.astype("float32") / 255

# defining our model, chain of two Dense layer
model = keras.Sequential([
 layers.Dense(512, activation="relu"),  # relu = max(x, 0)
 layers.Dense(10, activation="softmax") # softmax classification layer -> returns an array of 10 probability scores
])

# compilation step
model.compile(optimizer="rmsprop",  # mechanism to improve model's perfomace
 loss="sparse_categorical_crossentropy",    # loss function used as a feedbacksign for learning wegiht tensors (measure of the model's perfomance)
 metrics=["accuracy"])  # metrics to monitor accuracy 

# fitting the model to its training data, training loop
model.fit(train_images, train_labels, epochs=5, batch_size=128)
# here, the model iterates over the training data in batches of 128 samples, 5 times over (each iteration is called epoch)

# test the predictions
test_digits = test_images[0:10]
predictions = model.predict(test_digits) 
print(predictions[1].argmax())
print(predictions[1][2])
print(test_labels[1])