import keras
from keras.layers import Input, Conv2D, LayerNormalization, Dense, Concatenate

def dl_model():  
    input_layer = Input(shape=(32, 32, 3))
    
    # Depthwise Separable Convolution
    depthwise_conv = Conv2D(filters=32, kernel_size=(7, 7), strides=(2, 2), padding='same', depth_multiplier=1, activation='relu')(input_layer)
    depthwise_norm = LayerNormalization()(depthwise_conv)
    
    # Fully Connected Layers for Channel-wise Feature Transformation
    fc1 = Dense(units=32, activation='relu')(depthwise_norm)
    fc2 = Dense(units=32, activation='relu')(fc1)

    # Combine Original Input with Processed Features
    combined_features = Concatenate()([input_layer, depthwise_norm])

    # Classification Layers
    flatten_layer = keras.layers.Flatten()(combined_features)
    dense3 = Dense(units=128, activation='relu')(flatten_layer)
    output_layer = Dense(units=10, activation='softmax')(dense3)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model