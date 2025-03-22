import keras
from keras.layers import Input, Conv2D, Softmax, Lambda, LayerNormalization, ReLU
from keras.layers import Flatten, Dense

def dl_model():     

    input_layer = Input(shape=(32, 32, 3))

    # Generate attention weights
    attention_conv = Conv2D(filters=1, kernel_size=(1, 1), activation='softmax')(input_layer)
    attention_weights = Lambda(lambda x: x * attention_conv)(input_layer)

    # Reduce dimensionality
    dim_reduction_conv = Conv2D(filters=input_layer.shape[-1] // 3, kernel_size=(1, 1))(attention_weights)
    norm_layer = LayerNormalization()(dim_reduction_conv)
    relu_layer = ReLU()(norm_layer)

    # Restore dimensionality
    restore_conv = Conv2D(filters=input_layer.shape[-1], kernel_size=(1, 1))(relu_layer)

    # Add contextual information
    output_tensor = add([restore_conv, attention_weights])

    # Classification
    flatten_layer = Flatten()(output_tensor)
    output_layer = Dense(units=10, activation='softmax')(flatten_layer)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model