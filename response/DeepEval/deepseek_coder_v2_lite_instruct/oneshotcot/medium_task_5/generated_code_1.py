import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Add, BatchNormalization, Flatten, Dense

def dl_model():
    # Define the input layer
    input_layer = Input(shape=(32, 32, 3))
    
    # Main path
    def main_path(input_tensor):
        # First block
        conv1 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        conv2 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(conv1)
        max_pool1 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(conv2)
        
        # Second block
        conv3 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(max_pool1)
        conv4 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(conv3)
        max_pool2 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(conv4)
        
        return max_pool2
    
    # Branch path
    def branch_path(input_tensor):
        conv5 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        max_pool3 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(conv5)
        return max_pool3
    
    # Apply paths
    main_output = main_path(input_layer)
    branch_output = branch_path(input_layer)
    
    # Add paths
    added_output = Add()([main_output, branch_output])
    
    # Batch normalization and flattening
    batch_norm = BatchNormalization()(added_output)
    flatten_layer = Flatten()(batch_norm)
    
    # Fully connected layers
    dense1 = Dense(units=256, activation='relu')(flatten_layer)
    dense2 = Dense(units=128, activation='relu')(dense1)
    output_layer = Dense(units=10, activation='softmax')(dense2)
    
    # Construct the model
    model = keras.Model(inputs=input_layer, outputs=output_layer)
    
    return model