import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, Flatten, Dense

def dl_model():
    input_layer = Input(shape=(32, 32, 3))  # CIFAR-10 images are 32x32 with 3 color channels
    
    # Initial 1x1 convolutional layer
    initial_conv = Conv2D(filters=32, kernel_size=(1, 1), activation='relu')(input_layer)
    
    # Branch 1: Local features through a 3x3 convolutional layer
    branch1 = Conv2D(filters=32, kernel_size=(3, 3), activation='relu')(initial_conv)
    
    # Branch 2: Sequential downsampling and upsampling with max pooling and 3x3 convolutional layer
    branch2 = MaxPooling2D(pool_size=(2, 2), strides=2)(initial_conv)
    branch2 = Conv2D(filters=32, kernel_size=(3, 3), activation='relu')(branch2)
    branch2 = MaxPooling2D(pool_size=(2, 2), strides=2)(branch2)
    branch2 = Conv2D(filters=32, kernel_size=(3, 3), activation='relu')(branch2)
    branch2 = MaxPooling2D(pool_size=(2, 2), strides=2)(branch2)
    
    # Branch 3: Sequential downsampling and upsampling with max pooling and 3x3 convolutional layer
    branch3 = MaxPooling2D(pool_size=(2, 2), strides=2)(initial_conv)
    branch3 = Conv2D(filters=32, kernel_size=(3, 3), activation='relu')(branch3)
    branch3 = MaxPooling2D(pool_size=(2, 2), strides=2)(branch3)
    branch3 = Conv2D(filters=32, kernel_size=(3, 3), activation='relu')(branch3)
    branch3 = MaxPooling2D(pool_size=(2, 2), strides=2)(branch3)
    
    # Concatenate outputs of all branches
    concatenated = Concatenate()([branch1, branch2, branch3])
    
    # Final 1x1 convolutional layer
    final_conv = Conv2D(filters=64, kernel_size=(1, 1), activation='relu')(concatenated)
    
    # Flatten the output and pass through fully connected layers
    flattened = Flatten()(final_conv)
    fc1 = Dense(units=128, activation='relu')(flattened)
    fc2 = Dense(units=64, activation='relu')(fc1)
    output_layer = Dense(units=10, activation='softmax')(fc2)
    
    model = keras.Model(inputs=input_layer, outputs=output_layer)
    
    return model