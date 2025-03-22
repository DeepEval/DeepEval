import keras
from keras.layers import Input, Conv2D, DepthwiseConv2D, GlobalAveragePooling2D, Dense, Reshape, Multiply, Add, Flatten

def dl_model():
    input_layer = Input(shape=(32, 32, 3))

    # Initial feature extraction
    conv1 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_layer)
    conv2 = DepthwiseConv2D(kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(conv1)

    # Channel attention mechanism
    gap = GlobalAveragePooling2D()(conv2)
    fc1 = Dense(units=64, activation='relu')(gap)
    fc2 = Dense(units=64, activation='sigmoid')(fc1)
    fc2 = Reshape(target_shape=(1, 1, 64))(fc2)
    ca_weights = Multiply()([fc2, conv2])

    # Channel attention weighting
    weighted_features = Multiply()([ca_weights, conv2])

    # Final feature reduction and classification
    conv3 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(weighted_features)
    output = Add()([conv3, input_layer])
    flatten = Flatten()(output)
    output_layer = Dense(units=10, activation='softmax')(flatten)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model