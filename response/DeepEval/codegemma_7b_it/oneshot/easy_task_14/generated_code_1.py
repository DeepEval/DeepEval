import keras
from keras.layers import Input, Conv2D, MaxPooling2D, GlobalAveragePooling2D, Dense, Reshape, Flatten, Multiply

def dl_model():
    input_layer = Input(shape=(32, 32, 3))

    # Compress the input features with global average pooling
    global_pool = GlobalAveragePooling2D()(input_layer)

    # Two fully connected layers to learn correlations among channels
    dense1 = Dense(units=32, activation='relu')(global_pool)
    dense2 = Dense(units=32, activation='relu')(dense1)

    # Reshape weights to match the input shape
    reshaped_dense2 = Reshape((32,))(dense2)

    # Element-wise multiplication with the input feature map
    weighted_feature_map = Multiply()([reshaped_dense2, input_layer])

    # Flatten the result and pass it through a final fully connected layer
    flatten = Flatten()(weighted_feature_map)
    output_layer = Dense(units=10, activation='softmax')(flatten)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model