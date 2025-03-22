import keras
from keras.layers import Input, AveragePooling2D, Flatten, Concatenate, Dense, Reshape, Conv2D, Dropout

def dl_model():
    input_layer = Input(shape=(28, 28, 1))
    
    # First block: Average Pooling and Flatten
    avg_pool1 = AveragePooling2D(pool_size=(1, 1), strides=1, padding='same')(input_layer)
    avg_pool2 = AveragePooling2D(pool_size=(2, 2), strides=2, padding='same')(input_layer)
    avg_pool3 = AveragePooling2D(pool_size=(4, 4), strides=4, padding='same')(input_layer)

    flat1 = Flatten()(avg_pool1)
    flat2 = Flatten()(avg_pool2)
    flat3 = Flatten()(avg_pool3)

    concatenated = Concatenate()([flat1, flat2, flat3])

    # Fully connected layer and reshape to 4D tensor
    fc1 = Dense(units=128, activation='relu')(concatenated)
    reshaped = Reshape((4, 4, 8))(fc1)  # Reshape based on desired dimensions

    # Second block: Multi-path Convolution and Dropout
    def multi_scale_block(input_tensor):
        path1 = Conv2D(filters=32, kernel_size=(1, 1), activation='relu', padding='same')(input_tensor)
        dropout1 = Dropout(0.5)(path1)
        
        path2 = Conv2D(filters=32, kernel_size=(1, 1), activation='relu', padding='same')(input_tensor)
        path2 = Conv2D(filters=32, kernel_size=(3, 3), activation='relu', padding='same')(path2)
        path2 = Conv2D(filters=32, kernel_size=(3, 3), activation='relu', padding='same')(path2)
        dropout2 = Dropout(0.5)(path2)
        
        path3 = Conv2D(filters=32, kernel_size=(1, 1), activation='relu', padding='same')(input_tensor)
        path3 = Conv2D(filters=32, kernel_size=(3, 3), activation='relu', padding='same')(path3)
        dropout3 = Dropout(0.5)(path3)
        
        path4 = AveragePooling2D(pool_size=(1, 1), strides=1, padding='same')(input_tensor)
        path4 = Conv2D(filters=32, kernel_size=(1, 1), activation='relu', padding='same')(path4)
        dropout4 = Dropout(0.5)(path4)

        output_tensor = Concatenate(axis=-1)([dropout1, dropout2, dropout3, dropout4])
        
        return output_tensor

    block_output = multi_scale_block(reshaped)

    # Fully connected layers for classification
    flatten_layer = Flatten()(block_output)
    dense1 = Dense(units=128, activation='relu')(flatten_layer)
    output_layer = Dense(units=10, activation='softmax')(dense1)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model