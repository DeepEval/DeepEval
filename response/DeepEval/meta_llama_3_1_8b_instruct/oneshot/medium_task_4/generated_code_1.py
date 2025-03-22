import keras
from keras.layers import Input, Conv2D, AveragePooling2D, Add, BatchNormalization, Flatten, Dense

def dl_model():
    
    input_layer = Input(shape=(32, 32, 3))
    
    # Path 1
    def block1(input_tensor):
        conv1 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        conv2 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(conv1)
        pool = AveragePooling2D(pool_size=(2, 2), strides=2, padding='same')(conv2)
        return pool
        
    block1_output = block1(input_layer)
    block1_output = block1(block1_output)
    
    # Path 2
    conv2 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_layer)
    
    # Combine the outputs of Path 1 and Path 2
    combined_output = Add()([block1_output, conv2])
    
    # Feature Extraction Pathway
    bath_norm = BatchNormalization()(combined_output)
    conv3 = Conv2D(filters=128, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(bath_norm)
    pool = AveragePooling2D(pool_size=(2, 2), strides=2, padding='same')(conv3)
    
    # Flatten the result
    flatten_layer = Flatten()(pool)
    
    # Classification Layer
    output_layer = Dense(units=10, activation='softmax')(flatten_layer)
    
    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model