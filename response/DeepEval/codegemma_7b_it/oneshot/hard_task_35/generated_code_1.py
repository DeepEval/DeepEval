import keras
from keras.layers import Input, Conv2D, MaxPooling2D, GlobalAveragePooling2D, Dense, Reshape, Multiply, Concatenate, Flatten

def residual_block(input_tensor, filters, kernel_size):
    conv1 = Conv2D(filters=filters, kernel_size=(kernel_size, kernel_size), strides=(1, 1), padding='same', activation='relu')(input_tensor)
    conv2 = Conv2D(filters=filters, kernel_size=(kernel_size, kernel_size), strides=(1, 1), padding='same', activation='relu')(conv1)

    # Global Average Pooling
    gap = GlobalAveragePooling2D()(conv2)

    # Fully connected layers to generate weights
    dense1 = Dense(units=filters)(gap)
    dense2 = Dense(units=filters)(dense1)

    # Reshape weights to match input shape
    weights = Reshape((1, 1, filters))(dense2)

    # Element-wise multiplication
    output = Multiply()([input_tensor, weights])

    return output

def dl_model():
    input_layer = Input(shape=(32, 32, 3))

    # Branch 1
    branch1 = residual_block(input_layer, filters=64, kernel_size=3)

    # Branch 2
    branch2 = residual_block(input_layer, filters=64, kernel_size=5)

    # Concatenate branches
    concat = Concatenate()([branch1, branch2])

    # Flatten and fully connected layers
    flatten = Flatten()(concat)
    dense = Dense(units=64, activation='relu')(flatten)
    output_layer = Dense(units=10, activation='softmax')(dense)

    model = keras.Model(inputs=input_layer, outputs=output_layer)
    return model