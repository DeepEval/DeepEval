import keras
from keras.layers import Input, GlobalAveragePooling2D, Dense, Reshape, Multiply, Flatten

def dl_model():  
    input_layer = Input(shape=(32, 32, 3))

    x = GlobalAveragePooling2D()(input_layer)

    # Two fully connected layers to learn channel correlations
    x = Dense(units=32, activation='relu')(x) 
    x = Dense(units=32, activation='relu')(x) 

    # Reshape weights to align with input shape
    channel_weights = Dense(units=3, activation='linear')(x)
    channel_weights = Reshape((32, 32, 3))(channel_weights)

    # Element-wise multiplication with input feature map
    x = Multiply()([input_layer, channel_weights])

    # Flatten and final dense layer
    x = Flatten()(x)
    output_layer = Dense(units=10, activation='softmax')(x)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model