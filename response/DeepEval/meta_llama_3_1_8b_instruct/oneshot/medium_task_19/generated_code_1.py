import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, BatchNormalization, Flatten, Dense

def dl_model():     

    input_layer = Input(shape=(32, 32, 3))
    
    # Branch 1: 1x1 convolution for dimensionality reduction
    conv1 = Conv2D(filters=16, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_layer)
    
    # Branch 2: 1x1 convolution followed by 3x3 convolution
    conv2 = Conv2D(filters=16, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_layer)
    conv3 = Conv2D(filters=16, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(conv2)
    
    # Branch 3: 1x1 convolution followed by 5x5 convolution
    conv4 = Conv2D(filters=16, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_layer)
    conv5 = Conv2D(filters=16, kernel_size=(5, 5), strides=(1, 1), padding='same', activation='relu')(conv4)
    
    # Branch 4: 3x3 max pooling followed by 1x1 convolution
    maxpool = MaxPooling2D(pool_size=(3, 3), strides=(3, 3), padding='valid')(input_layer)
    conv6 = Conv2D(filters=16, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(maxpool)
    
    # Concatenate the outputs of these branches
    output_tensor = Concatenate()([conv1, conv3, conv5, conv6])
    
    # Apply batch normalization
    bath_norm = BatchNormalization()(output_tensor)
    
    # Flatten the output
    flatten_layer = Flatten()(bath_norm)
    
    # Apply two fully connected layers
    dense1 = Dense(units=128, activation='relu')(flatten_layer)
    output_layer = Dense(units=10, activation='softmax')(dense1)
    
    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model