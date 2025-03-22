import keras
from keras.layers import Input, Conv2D, AveragePooling2D, Flatten, Dense, Reshape, Concatenate, Dropout

def dl_model():
    input_layer = Input(shape=(28, 28, 1))

    # First specialized block
    avg_pool1 = AveragePooling2D(pool_size=(1, 1), strides=(1, 1), padding='same')(input_layer)
    avg_pool2 = AveragePooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(input_layer)
    avg_pool3 = AveragePooling2D(pool_size=(4, 4), strides=(4, 4), padding='same')(input_layer)

    flat1 = Flatten()(avg_pool1)
    flat2 = Flatten()(avg_pool2)
    flat3 = Flatten()(avg_pool3)

    concatenated_vector = Concatenate()([flat1, flat2, flat3])
    
    fc = Dense(units=128, activation='relu')(concatenated_vector)
    reshaped = Reshape((4, 4, 8))(fc)  # Example reshape target, adjust based on dimensions

    # Second specialized block
    def block(input_tensor):
        # Path 1
        path1_conv1 = Conv2D(filters=32, kernel_size=(1, 1), padding='same', activation='relu')(input_tensor)
        path1_dropout = Dropout(0.5)(path1_conv1)
        
        # Path 2
        path2_conv1 = Conv2D(filters=32, kernel_size=(1, 1), padding='same', activation='relu')(input_tensor)
        path2_conv2 = Conv2D(filters=32, kernel_size=(3, 3), padding='same', activation='relu')(path2_conv1)
        path2_conv3 = Conv2D(filters=32, kernel_size=(3, 3), padding='same', activation='relu')(path2_conv2)
        path2_dropout = Dropout(0.5)(path2_conv3)
        
        # Path 3
        path3_conv1 = Conv2D(filters=32, kernel_size=(1, 1), padding='same', activation='relu')(input_tensor)
        path3_conv2 = Conv2D(filters=32, kernel_size=(3, 3), padding='same', activation='relu')(path3_conv1)
        path3_dropout = Dropout(0.5)(path3_conv2)
        
        # Path 4
        path4_avg_pool = AveragePooling2D(pool_size=(3, 3), strides=(1, 1), padding='same')(input_tensor)
        path4_conv = Conv2D(filters=32, kernel_size=(1, 1), padding='same', activation='relu')(path4_avg_pool)
        path4_dropout = Dropout(0.5)(path4_conv)

        # Concatenation
        output_tensor = Concatenate()([path1_dropout, path2_dropout, path3_dropout, path4_dropout])

        return output_tensor

    block_output = block(input_tensor=reshaped)
    
    # Fully connected layers for final classification
    flat = Flatten()(block_output)
    dense1 = Dense(units=64, activation='relu')(flat)
    output_layer = Dense(units=10, activation='softmax')(dense1)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model