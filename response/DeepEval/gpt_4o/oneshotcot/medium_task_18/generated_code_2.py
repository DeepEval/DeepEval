import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, Flatten, Dense
from keras.models import Model

def dl_model():
    input_layer = Input(shape=(32, 32, 3))  # CIFAR-10 images are 32x32 with 3 color channels

    # Define the multi-scale feature extraction block
    def multi_scale_block(input_tensor):
        # 1x1 convolution path
        path1 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        
        # 3x3 convolution path
        path2 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        
        # 5x5 convolution path
        path3 = Conv2D(filters=32, kernel_size=(5, 5), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        
        # 3x3 max pooling path
        path4 = MaxPooling2D(pool_size=(3, 3), strides=(1, 1), padding='same')(input_tensor)
        
        # Concatenate all paths
        output_tensor = Concatenate()([path1, path2, path3, path4])
        
        return output_tensor

    # Apply the multi-scale feature extraction block
    block_output = multi_scale_block(input_layer)
    
    # Flatten the output
    flatten_layer = Flatten()(block_output)
    
    # Fully connected layers
    dense1 = Dense(units=256, activation='relu')(flatten_layer)
    output_layer = Dense(units=10, activation='softmax')(dense1)  # CIFAR-10 has 10 classes
    
    # Build the model
    model = Model(inputs=input_layer, outputs=output_layer)
    
    return model