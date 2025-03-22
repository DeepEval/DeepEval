import tensorflow as tf
from tensorflow import keras

# Create a compiled Keras model
model = keras.Sequential([
    keras.layers.Dense(64, activation='relu', input_shape=(784,)),
    keras.layers.Dense(10, activation='softmax')
])

# Compile the Keras model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Create an Estimator from the compiled Keras model
keras_estimator = tf.keras.estimator.model_to_estimator(
    keras_model=model,
    model_dir='/tmp/keras_model',
    config=tf.estimator.RunConfig()
)

# Define the input function
def input_function(data, labels, batch_size=32, num_epochs=1):
    dataset = tf.data.Dataset.from_tensor_slices((data, labels))
    dataset = dataset.shuffle(buffer_size=len(data)).repeat(num_epochs).batch(batch_size)
    return dataset.make_one_shot_iterator().get_next()

# Train the Keras model using the Estimator
keras_estimator.train(input_fn=lambda: input_function(data, labels))

# Evaluate the Keras model using the Estimator
keras_estimator.evaluate(input_fn=lambda: input_function(data, labels, num_epochs=1))