import keras
from keras.layers import Input, Conv2D, GlobalAveragePooling2D, Dense, Reshape, Add

def dl_model():
    input_layer = Input(shape=(32, 32, 3))

    # Main path
    conv = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_layer)
    gap = GlobalAveragePooling2D()(conv)
    fc1 = Dense(units=conv.shape[3], activation='relu')(gap)
    fc2 = Dense(units=conv.shape[3], activation='sigmoid')(fc1)
    reshaped_weights = Reshape((conv.shape))(fc2)
    weighted_conv = Multiply()([conv, reshaped_weights])

    # Branch path
    branch_conv = Conv2D(filters=conv.shape[3], kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_layer)

    # Combine paths
    combined = Add()([weighted_conv, branch_conv])

    # Fully connected layers
    flatten_layer = Flatten()(combined)
    dense1 = Dense(units=256, activation='relu')(flatten_layer)
    dense2 = Dense(units=128, activation='relu')(dense1)
    output_layer = Dense(units=10, activation='softmax')(dense2)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model