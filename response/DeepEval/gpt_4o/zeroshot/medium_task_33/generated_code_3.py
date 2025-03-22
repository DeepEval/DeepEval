import tensorflow as tf
from tensorflow.keras.layers import Input, Lambda, SeparableConv2D, Flatten, Dense, Concatenate
from tensorflow.keras.models import Model
from tensorflow.keras.datasets import cifar10

def dl_model():
    # Define input shape
    input_shape = (32, 32, 3)  # CIFAR-10 image size with 3 color channels
    
    # Input layer
    inputs = Input(shape=input_shape)
    
    # Split the input into three groups by color channel
    split_channels = Lambda(lambda x: tf.split(x, num_or_size_splits=3, axis=-1))(inputs)
    
    # Define separable convolutional operations for each group
    conv_1x1 = SeparableConv2D(32, (1, 1), padding='same', activation='relu')(split_channels[0])
    conv_3x3 = SeparableConv2D(32, (3, 3), padding='same', activation='relu')(split_channels[1])
    conv_5x5 = SeparableConv2D(32, (5, 5), padding='same', activation='relu')(split_channels[2])
    
    # Flatten the outputs from separable convolutions
    flat_1x1 = Flatten()(conv_1x1)
    flat_3x3 = Flatten()(conv_3x3)
    flat_5x5 = Flatten()(conv_5x5)
    
    # Concatenate the flattened outputs
    concatenated = Concatenate()([flat_1x1, flat_3x3, flat_5x5])
    
    # Fully connected layers
    fc1 = Dense(128, activation='relu')(concatenated)
    fc2 = Dense(64, activation='relu')(fc1)
    fc3 = Dense(10, activation='softmax')(fc2)  # CIFAR-10 has 10 classes
    
    # Create the model
    model = Model(inputs=inputs, outputs=fc3)
    
    # Compile the model
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    
    return model

# Instantiate the model
model = dl_model()

# Print the model summary
model.summary()