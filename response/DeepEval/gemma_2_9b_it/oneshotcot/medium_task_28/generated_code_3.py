import keras
from keras.layers import Input, Conv2D, Flatten, Dense, Softmax, LayerNormalization, Add

def dl_model():     

    input_layer = Input(shape=(32, 32, 3))
    
    # Attention Mechanism
    attention_weights = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='softmax')(input_layer)
    contextual_features = input_layer * attention_weights

    # Dimensionality Reduction & Restoration
    compressed = Conv2D(filters=input_layer.shape[-1]//3, kernel_size=(1, 1), strides=(1, 1), padding='same')(contextual_features)
    normalized = LayerNormalization()(compressed)
    activated = keras.layers.ReLU()(normalized)
    restored = Conv2D(filters=input_layer.shape[-1], kernel_size=(1, 1), strides=(1, 1), padding='same')(activated)

    # Feature Fusion
    output = Add()([input_layer, restored])

    # Classification
    flatten_layer = Flatten()(output)
    dense_layer = Dense(units=10, activation='softmax')(flatten_layer)

    model = keras.Model(inputs=input_layer, outputs=dense_layer)
    
    return model