import keras
from keras.layers import Input, DepthwiseConv2D, Add, LayerNormalization, GlobalAveragePooling2D, Dense, Reshape

def dl_model():
    
    # Define the depthwise separable convolutional layer
    input_layer = Input(shape=(32, 32, 3))
    dw_conv = DepthwiseConv2D(kernel_size=(7, 7), strides=(1, 1), padding='same', activation='relu')(input_layer)
    layer_norm = LayerNormalization()(dw_conv)

    # Define the first fully connected layer with the same number of channels as the input layer
    conv1 = Conv2D(kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(layer_norm)
    reshaped_conv1 = Reshape((-1, 3))(conv1)
    
    # Define the second fully connected layer with the same number of channels as the input layer
    conv2 = Conv2D(kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(layer_norm)
    reshaped_conv2 = Reshape((-1, 3))(conv2)

    # Combine the original input with the processed features through an addition operation
    combined = Add()([input_layer, reshaped_conv1, reshaped_conv2])

    # Average pooling layer
    avg_pool = GlobalAveragePooling2D()(combined)

    # Define the first fully connected layer
    dense1 = Dense(units=128, activation='relu')(avg_pool)
    
    # Define the second fully connected layer
    dense2 = Dense(units=64, activation='relu')(dense1)

    # Define the output layer
    output_layer = Dense(units=10, activation='softmax')(dense2)

    # Build the model
    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model