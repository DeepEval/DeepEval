import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, BatchNormalization, Flatten, Dense

def dl_model():     

    input_layer = Input(shape=(32, 32, 3))
    
    # Define the four parallel convolutional paths
    def path1(input_tensor):
        conv1 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        return conv1
    
    def path2(input_tensor):
        conv1 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        conv2 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(conv1)
        conv3 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(conv2)
        return conv3
    
    def path3(input_tensor):
        conv1 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        conv2 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(conv1)
        return conv2
    
    def path4(input_tensor):
        max_pool = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='same')(input_tensor)
        conv = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(max_pool)
        return conv
    
    # Stack the four parallel convolutional paths
    path1_output = path1(input_layer)
    path2_output = path2(input_layer)
    path3_output = path3(input_layer)
    path4_output = path4(input_layer)
    
    # Concatenate the outputs of the four paths
    concatenated_output = Concatenate()([path1_output, path2_output, path3_output, path4_output])
    
    # Apply batch normalization and flatten the result
    bath_norm = BatchNormalization()(concatenated_output)
    flatten_layer = Flatten()(bath_norm)
    
    # Pass the flattened result through a dense layer with 128 units
    dense1 = Dense(units=128, activation='relu')(flatten_layer)
    
    # Add the final output layer with softmax activation
    output_layer = Dense(units=10, activation='softmax')(dense1)
    
    # Create the Keras model
    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model