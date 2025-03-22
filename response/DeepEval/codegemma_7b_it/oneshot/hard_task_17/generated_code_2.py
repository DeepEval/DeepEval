import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, BatchNormalization, Flatten, Dense, GlobalAveragePooling2D, Reshape

def dl_model():

    input_layer = Input(shape=(32, 32, 3))
    conv = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='valid', activation='sigmoid')(input_layer)
    max_pooling = MaxPooling2D(pool_size=(2, 2), strides=2, padding='valid')(conv)

    # Block 1
    def block1(input_tensor):
        gap = GlobalAveragePooling2D()(input_tensor)
        fc1 = Dense(units=input_tensor.shape[3], activation='relu')(gap)
        fc2 = Dense(units=input_tensor.shape[3], activation='relu')(fc1)
        fc2_reshaped = Reshape((1, 1, input_tensor.shape[3]))(fc2)
        weighted_output = multiply([input_tensor, fc2_reshaped])
        return weighted_output

    weighted_output = block1(max_pooling)

    # Block 2
    conv1 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(weighted_output)
    conv2 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(conv1)
    maxpool = MaxPooling2D(pool_size=(2, 2), strides=2, padding='same')(conv2)

    # Branch from Block 1
    branch_output = block1(maxpool)

    # Fusion
    combined_output = add([maxpool, branch_output])

    # Fully connected layers
    flatten_layer = Flatten()(combined_output)
    dense1 = Dense(units=128, activation='relu')(flatten_layer)
    dense2 = Dense(units=64, activation='relu')(dense1)
    output_layer = Dense(units=10, activation='softmax')(dense2)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model