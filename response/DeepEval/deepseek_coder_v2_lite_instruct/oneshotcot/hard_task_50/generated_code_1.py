import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Dropout, Concatenate, BatchNormalization, Flatten, Dense, Lambda, SeparableConv2D
from tensorflow.keras.models import Model

def dl_model():
    input_layer = Input(shape=(32, 32, 3))

    # First block
    def first_block(input_tensor):
        pool1 = MaxPooling2D(pool_size=(1, 1), strides=1, padding='same')(input_tensor)
        pool2 = MaxPooling2D(pool_size=(2, 2), strides=2, padding='same')(input_tensor)
        pool3 = MaxPooling2D(pool_size=(4, 4), strides=4, padding='same')(input_tensor)
        
        # Flatten each pooled output
        flatten1 = Flatten()(pool1)
        flatten2 = Flatten()(pool2)
        flatten3 = Flatten()(pool3)
        
        # Apply dropout before concatenation
        dropout = Dropout(0.5)(tf.concat([flatten1, flatten2, flatten3], axis=1))
        
        # Transform the output to a 4D tensor
        reshape = Lambda(lambda x: tf.reshape(x, (-1, 1, 1, tf.shape(x)[1])))(dropout)
        return reshape

    first_block_output = first_block(input_layer)

    # Second block
    def second_block(input_tensor):
        # Split the input into four groups
        split_1x1 = Lambda(lambda x: tf.split(x, num_or_size_splits=4, axis=-1)[0])(input_tensor)
        split_3x3 = Lambda(lambda x: tf.split(x, num_or_size_splits=4, axis=-1)[1])(input_tensor)
        split_5x5 = Lambda(lambda x: tf.split(x, num_or_size_splits=4, axis=-1)[2])(input_tensor)
        split_7x7 = Lambda(lambda x: tf.split(x, num_or_size_splits=4, axis=-1)[3])(input_tensor)
        
        # Process each group with separable convolutional layers
        conv1x1 = SeparableConv2D(filters=32, kernel_size=(1, 1), padding='same', activation='relu')(split_1x1)
        conv3x3 = SeparableConv2D(filters=32, kernel_size=(3, 3), padding='same', activation='relu')(split_3x3)
        conv5x5 = SeparableConv2D(filters=32, kernel_size=(5, 5), padding='same', activation='relu')(split_5x5)
        conv7x7 = SeparableConv2D(filters=32, kernel_size=(7, 7), padding='same', activation='relu')(split_7x7)
        
        # Concatenate the outputs
        concatenated = Concatenate(axis=-1)([conv1x1, conv3x3, conv5x5, conv7x7])
        return concatenated

    second_block_output = second_block(first_block_output)

    # Flatten the output and pass through a fully connected layer
    flatten = Flatten()(second_block_output)
    dense = Dense(units=10, activation='softmax')(flatten)

    model = Model(inputs=input_layer, outputs=dense)
    return model

# Usage
model = dl_model()
model.summary()