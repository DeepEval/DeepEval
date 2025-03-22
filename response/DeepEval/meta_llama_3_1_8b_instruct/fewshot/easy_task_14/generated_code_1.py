import keras
from keras.layers import Input, GlobalAveragePooling2D, Dense, Reshape, Multiply, Flatten

def dl_model():
    
    input_layer = Input(shape=(32, 32, 3))

    # Compressing the input features with global average pooling
    pool_layer = GlobalAveragePooling2D()(input_layer)

    # Utilizing two fully connected layers to generate weights
    dense1 = Dense(units=64, activation='relu')(pool_layer)
    dense2 = Dense(units=64, activation='relu')(dense1)

    # Reshaping the weights to align with the input shape
    weight_layer = Dense(units=3 * 32 * 32, use_bias=False)(dense2)
    weight_layer = Reshape(target_shape=(3, 32, 32))(weight_layer)

    # Element-wise multiplication with the input feature map
    weighted_layer = Multiply()([input_layer, weight_layer])

    # Flatten the result
    flatten_layer = Flatten()(weighted_layer)

    # Output layer
    output_layer = Dense(units=10, activation='softmax')(flatten_layer)

    # Construct the model
    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model