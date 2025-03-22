import tensorflow as tf
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Lambda, Conv2D, MaxPooling2D, Flatten, Dense, Concatenate

# Load CIFAR-10 data
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

# Normalize pixel values to be between 0 and 1
x_train, x_test = x_train / 255.0, x_test / 255.0

# Split input along channel dimension
input_shape = x_train.shape[1:]
channels_dim = len(input_shape)
split_indices = [tf.Tensor(channels_dim // 3)]
output_channels = input_shape[0] // 3

input_tensor = Input(shape=input_shape)
x = Lambda(lambda tensors: tf.split(tensors, split_indices, axis=channels_dim))(input_tensor)

# Apply 1x1 convolutions to each group
x = Conv2D(output_channels, (1, 1), padding='same')(x[0])
x = Conv2D(output_channels, (1, 1), padding='same')(x[1])
x = Conv2D(output_channels, (1, 1), padding='same')(x[2])

# Average pooling
x = MaxPooling2D(pool_size=(2, 2))(x)

# Concatenate along the channel dimension
x = Concatenate(axis=channels_dim)(x)

# Flatten and pass through two fully connected layers
x = Flatten()(x)
x = Dense(512, activation='relu')(x)
x = Dense(10, activation='softmax')(x)  # Assuming 10 classes for CIFAR-10

# Create the model
model = Model(inputs=input_tensor, outputs=x)

# Compile the model
model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

# Print a summary of the model
model.summary()

return model