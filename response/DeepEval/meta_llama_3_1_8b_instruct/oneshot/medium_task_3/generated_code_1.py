import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Add, Dense

def dl_model():
    
    input_layer = Input(shape=(28, 28, 1))
    
    def block(input_tensor):
        conv1 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        max_pooling = MaxPooling2D(pool_size=(2, 2), strides=2, padding='same')(conv1)
        
        return max_pooling
    
    block1_output = block(input_layer)
    block2_output = block(block1_output)
    
    # Combine the outputs from both paths
    combined_output = Add()([input_layer, block1_output, block2_output])
    
    flatten_layer = Flatten()(combined_output)
    dense_layer = Dense(units=10, activation='softmax')(flatten_layer)
    
    model = keras.Model(inputs=input_layer, outputs=dense_layer)
    
    return model