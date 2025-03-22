from tensorflow.keras.layers import Input, Lambda, Conv2D, Concatenate, Flatten, Dense
from tensorflow.keras.models import Model
import tensorflow as tf

def dl_model():
    # Define the input shape based on CIFAR-10 dataset
    input_shape = (32, 32, 3)
    num_classes = 10

    # Create the input layer
    inputs = Input(shape=input_shape)

    # Split the input into three groups along the channel dimension
    split_inputs = Lambda(lambda x: tf.split(x, num_or_size_splits=3, axis=-1))(inputs)

    # Apply different convolutional kernels to each group
    conv1x1 = Conv2D(32, (1, 1), activation='relu', padding='same')(split_inputs[0])
    conv3x3 = Conv2D(32, (3, 3), activation='relu', padding='same')(split_inputs[1])
    conv5x5 = Conv2D(32, (5, 5), activation='relu', padding='same')(split_inputs[2])

    # Concatenate the feature maps from different convolutional operations
    concatenated = Concatenate(axis=-1)([conv1x1, conv3x3, conv5x5])

    # Flatten the concatenated feature maps
    flattened = Flatten()(concatenated)

    # Fully connected layers
    fc1 = Dense(128, activation='relu')(flattened)
    fc2 = Dense(64, activation='relu')(fc1)
    
    # Output layer
    outputs = Dense(num_classes, activation='softmax')(fc2)

    # Create the model
    model = Model(inputs=inputs, outputs=outputs)

    return model

# Instantiate and compile the model
model = dl_model()
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Summary of the model
model.summary()