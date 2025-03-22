import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, BatchNormalization, Flatten, Dense

# Define the input shape
input_shape = (32, 32, 3)

# Define the number of classes
num_classes = 10

# Define the model architecture
model = keras.Sequential([
    # First branch
    Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='valid', activation='relu')(input_shape),
    Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='valid', activation='relu')(input_shape),
    MaxPooling2D(pool_size=(2, 2), strides=2, padding='valid')(input_shape),
    # Second branch
    Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='valid', activation='relu')(input_shape),
    Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='valid', activation='relu')(input_shape),
    Conv2D(filters=64, kernel_size=(5, 5), strides=(1, 1), padding='valid', activation='relu')(input_shape),
    MaxPooling2D(pool_size=(2, 2), strides=2, padding='valid')(input_shape),
    # Third branch
    MaxPooling2D(pool_size=(2, 2), strides=2, padding='valid')(input_shape),
    # Concatenate the outputs from all branches
    Concatenate()([
        # First branch
        model.get_layer(index=0).output,
        model.get_layer(index=1).output,
        model.get_layer(index=2).output,
        # Second branch
        model.get_layer(index=3).output,
        model.get_layer(index=4).output,
        model.get_layer(index=5).output,
        # Third branch
        model.get_layer(index=6).output,
    ]),
    # Flatten the output and add a dense layer
    Flatten()(model.get_layer(index=7).output),
    Dense(units=128, activation='relu')(model.get_layer(index=7).output),
    Dense(units=num_classes, activation='softmax')(model.get_layer(index=7).output),
])

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])