import keras
from keras.models import Model
from keras.layers import Input, Conv2D, MaxPooling2D, AveragePooling2D, Concatenate, BatchNormalization, Flatten, Dense

def dl_model():
    # Input layer
    input_layer = Input(shape=(32, 32, 3))
    
    # Path 1: Convolutional blocks and average pooling
    def path1(input_tensor):
        conv1 = Conv2D(filters=32, kernel_size=(3, 3), padding='same', activation='relu')(input_tensor)
        bn1 = BatchNormalization()(conv1)
        avg_pool1 = AveragePooling2D(pool_size=(2, 2))(bn1)
        
        conv2 = Conv2D(filters=64, kernel_size=(3, 3), padding='same', activation='relu')(avg_pool1)
        bn2 = BatchNormalization()(conv2)
        avg_pool2 = AveragePooling2D(pool_size=(2, 2), strides=(2, 2))(bn2)

        return avg_pool2
    
    path1_output = path1(input_layer)
    
    # Path 2: Single convolutional layer
    def path2(input_tensor):
        conv = Conv2D(filters=32, kernel_size=(3, 3), padding='same', activation='relu')(input_tensor)
        return conv
    
    path2_output = path2(input_layer)
    path2_output = path2(path1_output)
    
    # Combine outputs from both paths
    combined_output = Concatenate()([path1_output, path2_output])
    
    # Flatten and fully connected layers
    flatten = Flatten()(combined_output)
    dense1 = Dense(units=256, activation='relu')(flatten)
    dense2 = Dense(units=128, activation='relu')(dense1)
    output_layer = Dense(units=10, activation='softmax')(dense2)
    
    # Construct the model
    model = Model(inputs=input_layer, outputs=output_layer)
    
    return model