import keras
from keras.layers import Input, Conv2D, LayerNormalization, Dense, Concatenate

def dl_model():     

    input_layer = Input(shape=(32, 32, 3))
    
    # 7x7 Depthwise Separable Convolution
    depthwise = Conv2D(filters=32, kernel_size=(7, 7), strides=(2, 2), padding='same', depth_multiplier=1, activation='relu')(input_layer)
    norm_layer = LayerNormalization()(depthwise) 

    # Fully Connected Layers for Channel-wise Feature Transformation
    fc1 = Dense(units=32, activation='relu')(norm_layer)
    fc2 = Dense(units=32, activation='relu')(fc1)

    # Combine Original Input and Processed Features
    combined_features = Concatenate()([input_layer, fc2])

    # Final Classification Layers
    dense3 = Dense(units=10, activation='softmax')(combined_features) 

    model = keras.Model(inputs=input_layer, outputs=dense3)

    return model