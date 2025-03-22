import keras
from keras.layers import Input, MaxPooling2D, Flatten, Dropout, Dense, Lambda, Concatenate
from keras.layers import SeparableConv2D
import tensorflow as tf

def dl_model():
    input_layer = Input(shape=(32, 32, 3))  # CIFAR-10 images are 32x32 pixels with 3 color channels

    # First block with max pooling layers of different scales
    max_pool1 = MaxPooling2D(pool_size=(1, 1), strides=(1, 1), padding='same')(input_layer)
    max_pool2 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(input_layer)
    max_pool3 = MaxPooling2D(pool_size=(4, 4), strides=(4, 4), padding='same')(input_layer)

    # Flatten the outputs
    flat1 = Flatten()(max_pool1)
    flat2 = Flatten()(max_pool2)
    flat3 = Flatten()(max_pool3)

    # Apply Dropout to mitigate overfitting
    drop1 = Dropout(0.5)(flat1)
    drop2 = Dropout(0.5)(flat2)
    drop3 = Dropout(0.5)(flat3)

    # Concatenate flattened vectors
    concatenated = Concatenate()([drop1, drop2, drop3])

    # Fully connected layer
    dense1 = Dense(units=512, activation='relu')(concatenated)

    # Reshape to prepare for the second block
    reshaped = tf.reshape(dense1, (-1, 1, 1, 512))  # Reshape to (batch_size, 1, 1, 512)

    # Second block with separable convolutions
    split_groups = Lambda(lambda x: tf.split(x, num_or_size_splits=4, axis=-1))(reshaped)

    # Define convolutional layers for each group
    conv1 = SeparableConv2D(filters=64, kernel_size=(1, 1), padding='same', activation='relu')(split_groups[0])
    conv2 = SeparableConv2D(filters=64, kernel_size=(3, 3), padding='same', activation='relu')(split_groups[1])
    conv3 = SeparableConv2D(filters=64, kernel_size=(5, 5), padding='same', activation='relu')(split_groups[2])
    conv4 = SeparableConv2D(filters=64, kernel_size=(7, 7), padding='same', activation='relu')(split_groups[3])

    # Concatenate the outputs from the separable convolutions
    concatenated_conv = Concatenate()([conv1, conv2, conv3, conv4])

    # Flatten the concatenated output
    flatten_output = Flatten()(concatenated_conv)

    # Final fully connected layer for classification
    output_layer = Dense(units=10, activation='softmax')(flatten_output)  # 10 classes for CIFAR-10

    # Create the model
    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model