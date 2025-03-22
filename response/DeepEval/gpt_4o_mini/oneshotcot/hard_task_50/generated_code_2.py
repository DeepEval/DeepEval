import keras
from keras.layers import Input, MaxPooling2D, Flatten, Dropout, Dense, Reshape, Lambda, Concatenate
from keras.layers import SeparableConv2D
import tensorflow as tf

def dl_model():
    input_layer = Input(shape=(32, 32, 3))  # CIFAR-10 images are 32x32 RGB images

    # First block
    # Max pooling with different scales
    max_pool1 = MaxPooling2D(pool_size=(1, 1), strides=(1, 1), padding='same')(input_layer)
    max_pool2 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(input_layer)
    max_pool3 = MaxPooling2D(pool_size=(4, 4), strides=(4, 4), padding='same')(input_layer)

    # Flatten each pooling output
    flatten1 = Flatten()(max_pool1)
    flatten2 = Flatten()(max_pool2)
    flatten3 = Flatten()(max_pool3)

    # Apply Dropout
    dropout1 = Dropout(0.5)(flatten1)
    dropout2 = Dropout(0.5)(flatten2)
    dropout3 = Dropout(0.5)(flatten3)

    # Concatenate the flattened outputs
    concatenated = Concatenate()([dropout1, dropout2, dropout3])

    # Fully connected layer
    dense1 = Dense(units=512, activation='relu')(concatenated)

    # Reshape the output to 4D tensor for the second block
    reshape = Reshape((1, 1, 512))(dense1)  # Reshape for the next block

    # Second block
    split = Lambda(lambda x: tf.split(x, num_or_size_splits=4, axis=-1))(reshape)

    # Separable convolutions with different kernel sizes
    sep_conv1 = SeparableConv2D(filters=32, kernel_size=(1, 1), padding='same', activation='relu')(split[0])
    sep_conv2 = SeparableConv2D(filters=32, kernel_size=(3, 3), padding='same', activation='relu')(split[1])
    sep_conv3 = SeparableConv2D(filters=32, kernel_size=(5, 5), padding='same', activation='relu')(split[2])
    sep_conv4 = SeparableConv2D(filters=32, kernel_size=(7, 7), padding='same', activation='relu')(split[3])

    # Concatenate the outputs from separable convolutions
    concat_sep_conv = Concatenate()([sep_conv1, sep_conv2, sep_conv3, sep_conv4])

    # Flatten the output and add a final fully connected layer for classification
    flatten_final = Flatten()(concat_sep_conv)
    output_layer = Dense(units=10, activation='softmax')(flatten_final)  # CIFAR-10 has 10 classes

    # Build the model
    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model