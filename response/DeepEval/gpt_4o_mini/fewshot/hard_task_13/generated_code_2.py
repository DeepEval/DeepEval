import keras
import tensorflow as tf
from keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, GlobalAveragePooling2D, Dense, Reshape, Multiply

def dl_model():

    input_layer = Input(shape=(32, 32, 3))  # CIFAR-10 images are 32x32 RGB images

    # First Block
    def block_1(input_tensor):
        path1 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        path2 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        path3 = Conv2D(filters=32, kernel_size=(5, 5), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        path4 = MaxPooling2D(pool_size=(3, 3), strides=(1, 1), padding='same')(input_tensor)
        
        output_tensor = Concatenate()([path1, path2, path3, path4])
        return output_tensor

    block1_output = block_1(input_layer)

    # Second Block
    global_avg_pooling = GlobalAveragePooling2D()(block1_output)
    dense1 = Dense(units=128, activation='relu')(global_avg_pooling)
    dense2 = Dense(units=block1_output.shape[-1])(dense1)  # Size same as channels of block 2's input
    reshaped_weights = Reshape(target_shape=(1, 1, block1_output.shape[-1]))(dense2)  # Reshape to match input's shape
    
    # Element-wise multiplication
    weighted_output = Multiply()([block1_output, reshaped_weights])

    # Final Fully Connected Layer
    flatten = GlobalAveragePooling2D()(weighted_output)
    output_layer = Dense(units=10, activation='softmax')(flatten)  # CIFAR-10 has 10 classes

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model