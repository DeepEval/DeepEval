import keras
import tensorflow as tf
from keras.layers import Input, MaxPooling2D, Flatten, Concatenate, Dense, Reshape, Lambda, SeparableConv2D, Dropout

def dl_model():

    input_layer = Input(shape=(32, 32, 3))

    def block_1(input_tensor):
        # Three max pooling layers with different scales
        maxpool1 = MaxPooling2D(pool_size=(1, 1), strides=(1, 1), padding='same')(input_tensor)
        flatten1 = Flatten()(maxpool1)
        maxpool2 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(input_tensor)
        flatten2 = Flatten()(maxpool2)
        maxpool3 = MaxPooling2D(pool_size=(4, 4), strides=(4, 4), padding='same')(input_tensor)
        flatten3 = Flatten()(maxpool3)
        # Apply dropout to mitigate overfitting
        dropout1 = Dropout(rate=0.5)(flatten1)
        dropout2 = Dropout(rate=0.5)(flatten2)
        dropout3 = Dropout(rate=0.5)(flatten3)
        # Concatenate the flattened vectors
        output_tensor = Concatenate()([dropout1, dropout2, dropout3])
        return output_tensor

    def block_2(input_tensor):
        # Split input into four groups
        inputs_groups = Lambda(lambda x: tf.split(value=x, num_or_size_splits=4, axis=-1))(input_tensor)
        # Process each group with separable convolutional layers
        conv1 = SeparableConv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(inputs_groups[0])
        conv2 = SeparableConv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(inputs_groups[1])
        conv3 = SeparableConv2D(filters=32, kernel_size=(5, 5), strides=(1, 1), padding='same', activation='relu')(inputs_groups[2])
        conv4 = SeparableConv2D(filters=32, kernel_size=(7, 7), strides=(1, 1), padding='same', activation='relu')(inputs_groups[3])
        # Concatenate the outputs
        output_tensor = Concatenate()([conv1, conv2, conv3, conv4])
        return output_tensor

    # Build the model
    block1_output = block_1(input_tensor=input_layer)
    dense = Dense(units=256, activation='relu')(block1_output)
    reshaped = Reshape(target_shape=(8, 8, 4))(dense)
    block2_output = block_2(input_tensor=reshaped)
    flatten = Flatten()(block2_output)
    output_layer = Dense(units=10, activation='softmax')(flatten)
    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model