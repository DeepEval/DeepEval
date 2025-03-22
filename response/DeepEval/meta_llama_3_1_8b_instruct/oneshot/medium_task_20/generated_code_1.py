import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, BatchNormalization, Flatten, Dense

def dl_model():     

    input_layer = Input(shape=(32, 32, 3))
    
    # First path: 1x1 convolution
    conv1 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_layer)
    
    # Second path: 1x1 convolution -> 3x3 convolution -> 3x3 convolution
    conv2 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_layer)
    conv3 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(conv2)
    conv4 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(conv3)
    
    # Third path: 1x1 convolution -> 3x3 convolution
    conv5 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_layer)
    conv6 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(conv5)
    
    # Fourth path: max pooling -> 1x1 convolution
    maxpool = MaxPooling2D(pool_size=(2, 2), strides=2, padding='same')(input_layer)
    conv7 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(maxpool)
    
    # Concatenate the outputs from these paths
    output_tensor = Concatenate()([conv1, conv4, conv6, conv7])
    
    # Apply batch normalization and flatten the result
    bath_norm = BatchNormalization()(output_tensor)
    flatten_layer = Flatten()(bath_norm)
    
    # Add a dense layer with 128 units before the output layer
    dense_layer = Dense(units=128, activation='relu')(flatten_layer)
    
    # Final output layer with softmax activation
    output_layer = Dense(units=10, activation='softmax')(dense_layer)
    
    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model