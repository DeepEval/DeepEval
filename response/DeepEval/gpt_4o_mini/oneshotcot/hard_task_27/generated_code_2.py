import keras
from keras.layers import Input, DepthwiseConv2D, LayerNormalization, Dense, Add, Flatten
from keras.models import Model

def dl_model():
    input_layer = Input(shape=(32, 32, 3))  # CIFAR-10 images are 32x32 with 3 color channels

    # Step 1: Depthwise separable convolution
    depthwise_conv = DepthwiseConv2D(kernel_size=(7, 7), padding='same', activation='relu')(input_layer)
    
    # Step 2: Layer normalization
    layer_norm = LayerNormalization()(depthwise_conv)

    # Step 3: Fully connected layers
    dense1 = Dense(units=32, activation='relu')(layer_norm)  # Channels match the input layer
    dense2 = Dense(units=32, activation='relu')(dense1)  # Channels match the input layer

    # Step 4: Combine original input with processed features
    combined = Add()([input_layer, dense2])

    # Step 5: Final classification layers
    flatten_layer = Flatten()(combined)
    output_layer = Dense(units=128, activation='relu')(flatten_layer)
    final_output = Dense(units=10, activation='softmax')(output_layer)

    # Construct the model
    model = Model(inputs=input_layer, outputs=final_output)

    return model

# Example of how to create the model
model = dl_model()
model.summary()  # This will print the summary of the model