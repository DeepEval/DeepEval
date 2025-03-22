import keras
from keras.layers import Input, Conv2D, BatchNormalization, ReLU, Add, Flatten, Dense

def dl_model():
    
    input_layer = Input(shape=(32, 32, 3))  # CIFAR-10 images are 32x32x3
    initial_conv = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_layer)
    
    def parallel_block(input_tensor):
        # Block 1
        conv1 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same')(input_tensor)
        norm1 = BatchNormalization()(conv1)
        act1 = ReLU()(norm1)
        
        # Block 2
        conv2 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same')(input_tensor)
        norm2 = BatchNormalization()(conv2)
        act2 = ReLU()(norm2)
        
        # Block 3
        conv3 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same')(input_tensor)
        norm3 = BatchNormalization()(conv3)
        act3 = ReLU()(norm3)
        
        # Adding the outputs of all blocks together with initial convolution output
        added_output = Add()([initial_conv, act1, act2, act3])
        
        return added_output
    
    block_output = parallel_block(input_tensor=initial_conv)
    flatten_layer = Flatten()(block_output)
    dense1 = Dense(units=128, activation='relu')(flatten_layer)
    dense2 = Dense(units=64, activation='relu')(dense1)
    output_layer = Dense(units=10, activation='softmax')(dense2)  # CIFAR-10 has 10 classes

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model