import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, BatchNormalization, Flatten, Dense, GlobalAveragePooling2D, Reshape, Multiply

def dl_model():     

    input_layer = Input(shape=(32, 32, 3))
    conv = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='valid', activation='sigmoid')(input_layer)
    max_pooling = MaxPooling2D(pool_size=(2, 2), strides=2, padding='valid')(conv)

    # Compress input features with global average pooling
    gap = GlobalAveragePooling2D()(max_pooling)

    # Two fully connected layers to generate weights
    dense1 = Dense(units=32, activation='relu')(gap)
    dense2 = Dense(units=32, activation='relu')(dense1)

    # Reshape weights to match input shape
    gap_reshape = Reshape((1, 1, 32))(dense2)

    # Element-wise multiplication with input feature map
    multiply = Multiply()([gap_reshape, max_pooling])

    # Flatten and pass through final fully connected layer
    flatten_layer = Flatten()(multiply)
    output_layer = Dense(units=10, activation='softmax')(flatten_layer)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model