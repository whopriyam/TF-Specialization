import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow import keras

class myCallback(tf.keras.callbacks.Callback):
	def on_epoch_end(self, epoch, logs={}):
		if(logs.get('loss')<0.35):
			print("Loss is low so stopping training")
			self.model.stop_training = True

callbacks = myCallback()
#Dataset
fashion_mnist = tf.keras.datasets.fashion_mnist
(train_images,train_labels) , (test_images, test_labels) = fashion_mnist.load_data()

#Normalization
train_images = train_images/255.0
test_images = test_images/255.0

#Neural Network
model = keras.Sequential([
	keras.layers.Flatten(input_shape=(28,28)),
	keras.layers.Dense(128, activation=tf.nn.relu),
	keras.layers.Dense(10, activation=tf.nn.softmax)
	])

#Compiling the model
model.compile(optimizer = tf.optimizers.Adam(),
	loss = tf.keras.losses.sparse_categorical_crossentropy)
#Testing the model
model.fit(train_images, train_labels, epochs=10, callbacks = [callbacks])

print(model.evaluate(test_images, test_labels))
