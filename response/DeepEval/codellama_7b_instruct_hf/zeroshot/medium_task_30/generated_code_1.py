import keras
from keras.layers import Input, Flatten, Dense, Conv2D, MaxPooling2D
from keras.models import Model

# Define the input shape
input_shape = (32, 32, 3)

# Define the pooling layers
pool_1 = MaxPooling2D(pool_size=(1, 1), strides=(1, 1), padding='same')
pool_2 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')
pool_4 = MaxPooling2D(pool_size=(4, 4), strides=(4, 4), padding='same')

# Define the convolutional layers
conv_1 = Conv2D(32, (3, 3), activation='relu')
conv_2 = Conv2D(64, (3, 3), activation='relu')
conv_3 = Conv2D(128, (3, 3), activation='relu')

# Define the flatten layers
flat_1 = Flatten()
flat_2 = Flatten()

# Define the fully connected layers
dense_1 = Dense(128, activation='relu')
dense_2 = Dense(10, activation='softmax')

# Define the model
model = keras.models.Sequential([
    Input(input_shape),
    conv_1,
    pool_1,
    conv_2,
    pool_2,
    conv_3,
    pool_4,
    flat_1,
    dense_1,
    flat_2,
    dense_2
])

# Compile the model
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Return the constructed model
return model