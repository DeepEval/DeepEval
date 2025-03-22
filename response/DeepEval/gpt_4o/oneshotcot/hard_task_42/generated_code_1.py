import keras
from keras.layers import Input, Conv2D, MaxPooling2D, AveragePooling2D, Flatten, Dense, Dropout, Concatenate, Reshape

def dl_model():
    input_layer = Input(shape=(28, 28, 1))
    
    # Block 1: Three parallel paths with different max pooling scales
    path1_b1 = MaxPooling2D(pool_size=(1, 1), strides=(1, 1), padding='same')(input_layer)
    path1_b1_flat = Flatten()(path1_b1)
    path1_b1_dropout = Dropout(0.5)(path1_b1_flat)
    
    path2_b1 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(input_layer)
    path2_b1_flat = Flatten()(path2_b1)
    path2_b1_dropout = Dropout(0.5)(path2_b1_flat)
    
    path3_b1 = MaxPooling2D(pool_size=(4, 4), strides=(4, 4), padding='same')(input_layer)
    path3_b1_flat = Flatten()(path3_b1)
    path3_b1_dropout = Dropout(0.5)(path3_b1_flat)
    
    block1_output = Concatenate()([path1_b1_dropout, path2_b1_dropout, path3_b1_dropout])
    
    # Fully connected layer and reshape
    fc1 = Dense(units=128, activation='relu')(block1_output)
    reshaped = Reshape((4, 4, 8))(fc1)  # Reshape according to the required dimensions for the next block
    
    # Block 2: Four parallel paths with different convolution and pooling strategies
    path1_b2 = Conv2D(filters=32, kernel_size=(1, 1), activation='relu', padding='same')(reshaped)
    
    path2_b2 = Conv2D(filters=32, kernel_size=(1, 1), activation='relu', padding='same')(reshaped)
    path2_b2 = Conv2D(filters=32, kernel_size=(1, 7), activation='relu', padding='same')(path2_b2)
    path2_b2 = Conv2D(filters=32, kernel_size=(7, 1), activation='relu', padding='same')(path2_b2)
    
    path3_b2 = Conv2D(filters=32, kernel_size=(1, 1), activation='relu', padding='same')(reshaped)
    path3_b2 = Conv2D(filters=32, kernel_size=(7, 1), activation='relu', padding='same')(path3_b2)
    path3_b2 = Conv2D(filters=32, kernel_size=(1, 7), activation='relu', padding='same')(path3_b2)
    path3_b2 = Conv2D(filters=32, kernel_size=(7, 1), activation='relu', padding='same')(path3_b2)
    path3_b2 = Conv2D(filters=32, kernel_size=(1, 7), activation='relu', padding='same')(path3_b2)
    
    path4_b2 = AveragePooling2D(pool_size=(3, 3), strides=(1, 1), padding='same')(reshaped)
    path4_b2 = Conv2D(filters=32, kernel_size=(1, 1), activation='relu', padding='same')(path4_b2)
    
    block2_output = Concatenate()([path1_b2, path2_b2, path3_b2, path4_b2])
    
    # Flatten and fully connected layers for final classification
    flatten_layer = Flatten()(block2_output)
    dense1 = Dense(units=256, activation='relu')(flatten_layer)
    output_layer = Dense(units=10, activation='softmax')(dense1)

    model = keras.Model(inputs=input_layer, outputs=output_layer)
    
    return model