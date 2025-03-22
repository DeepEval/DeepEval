import keras
from keras.layers import Input, Conv2D, AveragePooling2D, Add, BatchNormalization, Flatten, Dense

def dl_model():     

    input_layer = Input(shape=(32, 32, 3))
    
    # Path 1: Two blocks of convolution followed by average pooling
    block1_output = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_layer)
    block1_output = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(block1_output)
    block1_output = AveragePooling2D(pool_size=(2, 2), strides=2, padding='valid')(block1_output)
    
    block2_output = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(block1_output)
    block2_output = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(block2_output)
    block2_output = AveragePooling2D(pool_size=(2, 2), strides=2, padding='valid')(block2_output)
    
    # Path 2: Single convolutional layer
    path2_output = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_layer)
    
    # Combine the two pathways through addition
    combined_output = Add()([block2_output, path2_output])
    
    # Flatten the output
    flatten_layer = Flatten()(combined_output)
    
    # Output layer
    output_layer = Dense(units=10, activation='softmax')(flatten_layer)
    
    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model