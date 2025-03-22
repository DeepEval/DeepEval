import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, Flatten, Dense

def dl_model():
    input_layer = Input(shape=(32, 32, 3))
    
    # First Branch: 1x1 Convolution for dimensionality reduction
    branch1 = Conv2D(filters=32, kernel_size=(1, 1), padding='same', activation='relu')(input_layer)
    
    # Second Branch: 1x1 Convolution followed by 3x3 Convolution
    branch2_1 = Conv2D(filters=32, kernel_size=(1, 1), padding='same', activation='relu')(input_layer)
    branch2_2 = Conv2D(filters=64, kernel_size=(3, 3), padding='same', activation='relu')(branch2_1)
    
    # Third Branch: 1x1 Convolution followed by 5x5 Convolution
    branch3_1 = Conv2D(filters=32, kernel_size=(1, 1), padding='same', activation='relu')(input_layer)
    branch3_2 = Conv2D(filters=64, kernel_size=(5, 5), padding='same', activation='relu')(branch3_1)
    
    # Fourth Branch: 3x3 Max Pooling followed by 1x1 Convolution
    branch4_pool = MaxPooling2D(pool_size=(3, 3), strides=(1, 1), padding='same')(input_layer)
    branch4_conv = Conv2D(filters=32, kernel_size=(1, 1), padding='same', activation='relu')(branch4_pool)
    
    # Concatenate all branches
    concatenated_branches = Concatenate()([branch1, branch2_2, branch3_2, branch4_conv])
    
    # Flatten the features and apply two fully connected layers
    flatten_layer = Flatten()(concatenated_branches)
    dense1 = Dense(units=128, activation='relu')(flatten_layer)
    output_layer = Dense(units=10, activation='softmax')(dense1)
    
    # Define the model
    model = keras.Model(inputs=input_layer, outputs=output_layer)
    
    return model