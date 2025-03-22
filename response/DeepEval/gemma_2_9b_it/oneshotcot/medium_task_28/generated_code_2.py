import keras
from keras.layers import Input, Conv2D, Softmax, Lambda, LayerNormalization, ReLU, Flatten, Dense

def dl_model():
    
    input_layer = Input(shape=(32, 32, 3))

    # Attention Block
    attention = Conv2D(filters=1, kernel_size=(1, 1), activation='softmax')(input_layer)
    attention_weighted = Lambda(lambda x: x[0] * x[1])([input_layer, attention]) 

    # Dimensionality Reduction
    x = Conv2D(filters=input_layer.shape[-1] // 3, kernel_size=(1, 1))(attention_weighted)
    x = LayerNormalization()(x)
    x = ReLU()(x)

    # Dimensionality Restoration
    x = Conv2D(filters=input_layer.shape[-1], kernel_size=(1, 1))(x)

    # Add to Original Input
    x = keras.layers.add([input_layer, x])

    # Flatten and Classify
    x = Flatten()(x)
    output_layer = Dense(units=10, activation='softmax')(x)

    model = keras.Model(inputs=input_layer, outputs=output_layer)
    
    return model