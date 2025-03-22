import keras
from keras.layers import Input, Lambda, Conv2D, MaxPooling2D, Concatenate, BatchNormalization, Flatten, Dense, ZeroPadding2D

def dl_model():
    def depthwise_separable_block(input_tensor, num_filters, kernel_size):
        conv = Conv2D(num_filters, kernel_size=(kernel_size, kernel_size), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        conv = Conv2D(num_filters, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(conv)
        return conv

    def branch_block(input_tensor, num_filters):
        path1 = depthwise_separable_block(input_tensor, num_filters, 1)
        path2 = depthwise_separable_block(input_tensor, num_filters, 3)
        path3 = depthwise_separable_block(input_tensor, num_filters, 5)
        avg_pool = MaxPooling2D(pool_size=(1, 1), strides=(1, 1), padding='valid')(input_tensor)
        return Concatenate()(inputs=[path1, path2, path3, avg_pool])

    input_layer = Input(shape=(32, 32, 3))

    # First block: three groups for different feature extraction
    split_tensor = Lambda(lambda t: keras.backend.split(t, 3, axis=-1))(input_layer)
    output_tensor1 = branch_block(split_tensor[0], num_filters=64)
    output_tensor2 = branch_block(split_tensor[1], num_filters=64)
    output_tensor3 = branch_block(split_tensor[2], num_filters=64)

    # Concatenate the outputs of the first block
    concatenated_tensor = Concatenate(axis=-1)(inputs=[output_tensor1, output_tensor2, output_tensor3])

    # Second block: multiple branches for feature extraction
    branch1 = depthwise_separable_block(input_tensor=concatenated_tensor, num_filters=64, kernel_size=1)
    branch2 = depthwise_separable_block(input_tensor=concatenated_tensor, num_filters=64, kernel_size=3)
    branch3 = depthwise_separable_block(input_tensor=concatenated_tensor, num_filters=64, kernel_size=5)
    avg_pool = MaxPooling2D(pool_size=(1, 1), strides=(1, 1), padding='valid')(input_tensor)

    # Flatten and pass through fully connected layers
    flattened_tensor = Flatten()(concatenated_tensor)
    dense1 = Dense(units=128, activation='relu')(flattened_tensor)
    dense2 = Dense(units=64, activation='relu')(dense1)
    output_layer = Dense(units=10, activation='softmax')(dense2)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model

model = dl_model()
model.summary()