import keras
from keras.layers import Input, MaxPooling2D, Flatten, Dropout, Dense, Reshape, Lambda, Concatenate, SeparableConv2D
from keras.models import Model
import tensorflow as tf

def dl_model():
    
    input_layer = Input(shape=(32, 32, 3))  # CIFAR-10 images are 32x32 with 3 color channels

    # First block with max pooling layers
    max_pool_1x1 = MaxPooling2D(pool_size=(1, 1), strides=1, padding='same')(input_layer)
    max_pool_2x2 = MaxPooling2D(pool_size=(2, 2), strides=2, padding='same')(input_layer)
    max_pool_4x4 = MaxPooling2D(pool_size=(4, 4), strides=4, padding='same')(input_layer)

    flatten_1x1 = Flatten()(max_pool_1x1)
    flatten_2x2 = Flatten()(max_pool_2x2)
    flatten_4x4 = Flatten()(max_pool_4x4)

    # Apply dropout to mitigate overfitting
    dropout_1x1 = Dropout(0.5)(flatten_1x1)
    dropout_2x2 = Dropout(0.5)(flatten_2x2)
    dropout_4x4 = Dropout(0.5)(flatten_4x4)

    # Concatenate the dropout outputs
    concatenated = Concatenate()([dropout_1x1, dropout_2x2, dropout_4x4])

    # Fully connected layer
    dense_layer = Dense(units=512, activation='relu')(concatenated)

    # Reshape to four-dimensional tensor for the second block
    reshape_layer = Reshape((8, 8, 8))(dense_layer)  # Adjust dimensions as needed

    # Second block with separable convolutions
    def split_and_process(tensor):
        # Split the tensor into 4 groups along the last dimension
        splits = tf.split(tensor, num_or_size_splits=4, axis=-1)
        conv1x1 = SeparableConv2D(filters=32, kernel_size=(1, 1), padding='same', activation='relu')(splits[0])
        conv3x3 = SeparableConv2D(filters=32, kernel_size=(3, 3), padding='same', activation='relu')(splits[1])
        conv5x5 = SeparableConv2D(filters=32, kernel_size=(5, 5), padding='same', activation='relu')(splits[2])
        conv7x7 = SeparableConv2D(filters=32, kernel_size=(7, 7), padding='same', activation='relu')(splits[3])
        
        return Concatenate()([conv1x1, conv3x3, conv5x5, conv7x7])

    processed_splits = Lambda(split_and_process)(reshape_layer)

    # Flatten and final dense layer for classification
    flatten = Flatten()(processed_splits)
    output_layer = Dense(units=10, activation='softmax')(flatten)

    # Construct and return the model
    model = Model(inputs=input_layer, outputs=output_layer)

    return model