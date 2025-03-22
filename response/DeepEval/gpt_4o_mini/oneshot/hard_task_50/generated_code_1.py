import keras
from keras.layers import Input, MaxPooling2D, Flatten, Dropout, Dense, Lambda, Conv2D, Concatenate
import tensorflow as tf

def dl_model():
    input_layer = Input(shape=(32, 32, 3))  # CIFAR-10 images have shape 32x32 with 3 channels

    # First block with different max pooling scales
    max_pool1 = MaxPooling2D(pool_size=(1, 1), strides=(1, 1), padding='same')(input_layer)
    max_pool2 = MaxPooling2D(pool_size=(2, 2), strides=(1, 1), padding='same')(input_layer)
    max_pool3 = MaxPooling2D(pool_size=(4, 4), strides=(1, 1), padding='same')(input_layer)

    # Flatten the outputs of max pooling layers
    flatten1 = Flatten()(max_pool1)
    flatten2 = Flatten()(max_pool2)
    flatten3 = Flatten()(max_pool3)

    # Apply Dropout
    dropout1 = Dropout(0.5)(flatten1)
    dropout2 = Dropout(0.5)(flatten2)
    dropout3 = Dropout(0.5)(flatten3)

    # Concatenate the flattened vectors
    concatenated = Concatenate()([dropout1, dropout2, dropout3])
    
    # Fully connected layer
    dense1 = Dense(units=256, activation='relu')(concatenated)

    # Reshape to prepare for second block
    reshaped = tf.keras.layers.Reshape((1, 1, 256))(dense1)  # Reshape into a 4D tensor

    # Second block - split and apply separable convolutions
    split_tensor = Lambda(lambda x: tf.split(x, num_or_size_splits=4, axis=-1))(reshaped)
    
    # Apply separable convolutions with different kernel sizes
    conv1 = Conv2D(filters=64, kernel_size=(1, 1), padding='same', activation='relu')(split_tensor[0])
    conv2 = Conv2D(filters=64, kernel_size=(3, 3), padding='same', activation='relu')(split_tensor[1])
    conv3 = Conv2D(filters=64, kernel_size=(5, 5), padding='same', activation='relu')(split_tensor[2])
    conv4 = Conv2D(filters=64, kernel_size=(7, 7), padding='same', activation='relu')(split_tensor[3])

    # Concatenate the outputs of the convolutions
    block2_output = Concatenate()([conv1, conv2, conv3, conv4])

    # Flatten the output for the final classification
    final_flatten = Flatten()(block2_output)
    output_layer = Dense(units=10, activation='softmax')(final_flatten)  # 10 classes for CIFAR-10

    # Create the model
    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model