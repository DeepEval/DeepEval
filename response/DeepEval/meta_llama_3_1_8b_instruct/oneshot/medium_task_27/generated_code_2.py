import keras
from keras.layers import Input, Conv2D, Add, GlobalAveragePooling2D, Concatenate, BatchNormalization, Flatten, Dense

def dl_model():     

    input_layer = Input(shape=(32, 32, 3))
    
    # Convolutional Branch 1: 3x3 convolution
    conv1 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_layer)
    
    # Convolutional Branch 2: 5x5 convolution
    conv2 = Conv2D(filters=64, kernel_size=(5, 5), strides=(1, 1), padding='same', activation='relu')(input_layer)
    
    # Combine the outputs of the two convolutional branches
    output = Add()([conv1, conv2])
    
    # Global average pooling layer
    global_avg_pool = GlobalAveragePooling2D()(output)
    
    # Attention Weights
    dense1 = Dense(units=128, activation='relu')(global_avg_pool)
    attention_weights = Dense(units=10, activation='softmax')(dense1)
    
    # Weighted Output
    weighted_output = Multiply()([output, attention_weights])
    output = Lambda(lambda x: K.sum(x, axis=[1, 2], keepdims=True))(weighted_output)
    output = Flatten()(output)
    
    # Final Classification
    output = Dense(units=10, activation='softmax')(output)
    
    model = keras.Model(inputs=input_layer, outputs=output)

    return model