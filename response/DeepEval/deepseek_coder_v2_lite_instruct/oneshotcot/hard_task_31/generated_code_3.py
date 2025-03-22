import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Add, Dropout, BatchNormalization, Flatten, Dense, Lambda, SeparableConv2D
from keras.models import Model
import tensorflow as tf

def dl_model():
    input_layer = Input(shape=(32, 32, 3))

    # First block
    def main_branch(x):
        x = Conv2D(32, (3, 3), padding='same', activation='relu')(x)
        x = Conv2D(32, (1, 1), padding='same', activation='relu')(x)
        return x

    def branch_branch(x):
        return x

    main_path = main_branch(input_layer)
    branch_path = branch_branch(input_layer)
    added = Add()([main_path, branch_path])

    # Second block
    def separable_conv_block(x, kernel_size):
        x = SeparableConv2D(32, kernel_size, padding='same', activation='relu')(x)
        x = Dropout(0.5)(x)
        return x

    def split_and_concat(x):
        split_indices = [1, 1, 1]
        splits = tf.split(x, num_or_size_splits=split_indices, axis=-1)
        conv1x1 = separable_conv_block(splits[0], (1, 1))
        conv3x3 = separable_conv_block(splits[1], (3, 3))
        conv5x5 = separable_conv_block(splits[2], (5, 5))
        return tf.concat([conv1x1, conv3x3, conv5x5], axis=-1)

    second_block_output = split_and_concat(added)

    # Flatten and fully connected layers
    flatten_layer = Flatten()(second_block_output)
    dense_layer = Dense(128, activation='relu')(flatten_layer)
    output_layer = Dense(10, activation='softmax')(dense_layer)

    model = Model(inputs=input_layer, outputs=output_layer)
    return model

# Build and compile the model (optional, depending on the context)
model = dl_model()
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])