import keras
from keras.layers import Input, GlobalAveragePooling2D, Lambda, Dense, Reshape, Multiply, Flatten

def dl_model():

    input_layer = Input(shape=(32, 32, 3))

    # Global Average Pooling
    gavg_pool = GlobalAveragePooling2D()(input_layer)

    # Two fully connected layers to generate weights
    weights_1 = Dense(units=3, activation='relu')(gavg_pool)
    weights_1 = Dense(units=3, activation='relu')(weights_1)

    # Reshape the weights to align with the input shape
    weights_1 = Reshape(target_shape=(3, 3))(weights_1)

    # Element-wise multiplication of the weights with the input feature map
    element_wise_mul = Multiply()([input_layer, weights_1])

    # Flatten the result
    flattened = Flatten()(element_wise_mul)

    # Another fully connected layer to obtain the final probability distribution
    output_layer = Dense(units=10, activation='softmax')(flattened)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model