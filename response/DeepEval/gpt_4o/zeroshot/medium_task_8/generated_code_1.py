import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, Flatten, Dense, Lambda, Add, Concatenate
from tensorflow.keras.models import Model
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.utils import to_categorical

def dl_model():
    input_shape = (32, 32, 3)  # CIFAR-10 images have shape 32x32x3
    num_classes = 10

    # Input layer
    inputs = Input(shape=input_shape)

    # Split the input into three parts along the last dimension
    split_layer = Lambda(lambda x: tf.split(x, num_or_size_splits=3, axis=-1))(inputs)
    
    # Group 1: Remains unchanged
    group1 = split_layer[0]

    # Group 2: Pass through a 3x3 convolutional layer
    group2 = Conv2D(32, (3, 3), padding='same', activation='relu')(split_layer[1])

    # Combine Group 2 with Group 3
    combined = Concatenate(axis=-1)([group2, split_layer[2]])
    
    # Additional 3x3 convolution after combining
    combined_conv = Conv2D(32, (3, 3), padding='same', activation='relu')(combined)

    # Concatenate all three groups for the main path output
    main_path_output = Concatenate(axis=-1)([group1, group2, combined_conv])

    # Branch path with 1x1 convolution
    branch_path_output = Conv2D(32, (1, 1), padding='same', activation='relu')(inputs)

    # Fuse main and branch path outputs through addition
    fused_output = Add()([main_path_output, branch_path_output])

    # Flatten and add a dense layer for classification
    flattened = Flatten()(fused_output)
    outputs = Dense(num_classes, activation='softmax')(flattened)

    # Create the model
    model = Model(inputs=inputs, outputs=outputs)
    
    # Compile the model
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    return model

# Example usage:
# Load CIFAR-10 data
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

# Preprocess the data
x_train = x_train.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0
y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)

# Build the model
model = dl_model()

# Train the model
model.fit(x_train, y_train, batch_size=64, epochs=10, validation_split=0.1)

# Evaluate the model
model.evaluate(x_test, y_test)