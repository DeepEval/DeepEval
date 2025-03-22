import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Concatenate, Dense, Reshape

def dl_model():
    # First block
    input_layer = Input(shape=(28, 28, 1))
    conv1 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='valid', activation='relu')(input_layer)
    conv2 = Conv2D(filters=32, kernel_size=(2, 2), strides=(2, 2), padding='valid', activation='relu')(conv1)
    conv3 = Conv2D(filters=32, kernel_size=(4, 4), strides=(4, 4), padding='valid', activation='relu')(conv2)
    max_pooling = MaxPooling2D(pool_size=(1, 1), strides=(1, 1), padding='valid')(conv3)
    flat_layer = Flatten()(max_pooling)

    # Second block
    block_input = Reshape((1, 1, 32))(flat_layer)
    conv_branch1 = Conv2D(filters=16, kernel_size=(1, 1), strides=(1, 1), padding='valid', activation='relu')(block_input)
    conv_branch2 = Conv2D(filters=16, kernel_size=(3, 3), strides=(1, 1), padding='valid', activation='relu')(block_input)
    conv_branch3 = Conv2D(filters=16, kernel_size=(7, 7), strides=(1, 1), padding='valid', activation='relu')(block_input)
    avg_pooling = MaxPooling2D(pool_size=(7, 7), strides=(7, 7), padding='valid')(block_input)
    conv_branch4 = Conv2D(filters=16, kernel_size=(3, 3), strides=(1, 1), padding='valid', activation='relu')(block_input)
    concat_layer = Concatenate()([conv_branch1, conv_branch2, conv_branch3, conv_branch4, avg_pooling])
    dense_layer = Dense(units=128, activation='relu')(concat_layer)
    dense_layer = Dense(units=64, activation='relu')(dense_layer)
    output_layer = Dense(units=10, activation='softmax')(dense_layer)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model