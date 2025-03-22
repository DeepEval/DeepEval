import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Add, Flatten, Dense, GlobalAveragePooling2D

def dl_model():
    input_layer = Input(shape=(32, 32, 3))
    
    # First branch
    branch1 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_layer)
    branch1 = Conv2D(filters=64, kernel_size=(5, 5), strides=(1, 1), padding='same', activation='relu')(branch1)
    branch1 = MaxPooling2D(pool_size=(2, 2), strides=(1, 1), padding='same')(branch1)
    branch1 = Add()([branch1, input_layer])
    
    # Second branch
    branch2 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_layer)
    branch2 = Conv2D(filters=64, kernel_size=(5, 5), strides=(1, 1), padding='same', activation='relu')(branch2)
    branch2 = MaxPooling2D(pool_size=(2, 2), strides=(1, 1), padding='same')(branch2)
    branch2 = Add()([branch2, input_layer])
    
    # Combine branches
    concatenated_branches = Add()([branch1, branch2])
    
    # Global average pooling
    pooled_branches = GlobalAveragePooling2D()(concatenated_branches)
    
    # Fully connected layers
    flattened_branches = Flatten()(pooled_branches)
    fc1 = Dense(units=128, activation='relu')(flattened_branches)
    fc2 = Dense(units=64, activation='relu')(fc1)
    output_layer = Dense(units=10, activation='softmax')(fc2)
    
    model = keras.Model(inputs=input_layer, outputs=output_layer)
    
    return model