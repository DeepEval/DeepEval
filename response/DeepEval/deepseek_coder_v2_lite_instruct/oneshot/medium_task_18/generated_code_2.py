import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, BatchNormalization, Flatten, Dense

def dl_model():
    input_layer = Input(shape=(32, 32, 3))
    
    # First convolutional layer followed by max pooling
    conv1 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_layer)
    pool1 = MaxPooling2D(pool_size=(3, 3), strides=2, padding='same')(conv1)
    
    # Second convolutional layer followed by max pooling
    conv2 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(pool1)
    pool2 = MaxPooling2D(pool_size=(3, 3), strides=2, padding='same')(conv2)
    
    # Third convolutional layer followed by max pooling
    conv3 = Conv2D(filters=128, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(pool2)
    pool3 = MaxPooling2D(pool_size=(3, 3), strides=2, padding='same')(conv3)
    
    # Define the specific block with four parallel paths
    def block(input_tensor):
        path1 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        path2 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        path3 = Conv2D(filters=32, kernel_size=(5, 5), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        path4 = MaxPooling2D(pool_size=(3, 3), strides=(1, 1), padding='same')(input_tensor)
        output_tensor = Concatenate()([path1, path2, path3, path4])
        return output_tensor
    
    # Apply the block to the pool3 output
    block_output = block(input_tensor=pool3)
    
    # Batch normalization and flattening
    batch_norm = BatchNormalization()(block_output)
    flatten_layer = Flatten()(batch_norm)
    
    # Fully connected layers
    dense1 = Dense(units=256, activation='relu')(flatten_layer)
    dense2 = Dense(units=128, activation='relu')(dense1)
    output_layer = Dense(units=10, activation='softmax')(dense2)
    
    # Construct the model
    model = keras.Model(inputs=input_layer, outputs=output_layer)
    
    return model