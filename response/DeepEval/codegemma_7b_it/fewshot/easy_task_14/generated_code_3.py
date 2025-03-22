import keras
from keras.layers import Input, GlobalAveragePooling2D, Dense, Reshape, Multiply, Flatten

def dl_model():

    input_layer = Input(shape=(32, 32, 3))

    # Compress input features using global average pooling
    avg_pool = GlobalAveragePooling2D()(input_layer)

    # Two fully connected layers to generate weights
    dense1 = Dense(units=3 * 3 * 3, activation='relu')(avg_pool)
    dense2 = Dense(units=3 * 3 * 3, activation='softmax')(dense1)

    # Reshape weights to match input shape
    reshaped_dense2 = Reshape(target_shape=(3, 3, 3))(dense2)

    # Element-wise multiplication with input feature map
    multiply_output = Multiply()([input_layer, reshaped_dense2])

    # Flatten and pass through final fully connected layer
    flatten = Flatten()(multiply_output)
    output_layer = Dense(units=10, activation='softmax')(flatten)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model