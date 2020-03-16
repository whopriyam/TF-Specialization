import tensorflow as tf

data = tf.keras.datasets.fashion_mnist

(training_images,training_labels),(test_images,test_labels) = data.load_data()

training_images = training_images.reshape(60000,28,28,1)
training_images = training_images/255.0

test_images = test_images.reshape(10000,28,28,1)
test_images = test_images/255.0

model = tf.keras.models.Sequential([
	tf.keras.layers.Conv2D(64,(3,3), activation='relu', input_shape=(28,28,1)),
	tf.keras.layers.MaxPooling2D(2,2),
	tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
	tf.keras.layers.MaxPooling2D(2,2),
	tf.keras.layers.Flatten(),
	tf.keras.layers.Dense(128, activation='relu'),
	tf.keras.layers.Dense(10, activation='softmax')
	])

model.compile(optimizer=tf.keras.optimizers.Adam(), 
	loss=tf.keras.losses.sparse_categorical_crossentropy)

model.summary()

model.fit(training_images, training_labels, epochs=5)

test_loss = model.evaluate(test_images, test_labels)