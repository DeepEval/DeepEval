import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, Permute, Reshape, Concatenate

def dl_model():
    # Input layer
    input_layer = Input(shape=(32, 32, 3))
    
    # Reshape the input tensor to have three groups and reshape back
    reshaped_input = Permute((2, 3, 1))(input_layer)  # Change channels_last to channels_first
    reshaped_input = Reshape((32, 32, 3))(reshaped_input)  # Reshape back to (height, width, channels)
    
    # Swap third and fourth dimensions for channel shuffling
    shuffled_input = Permute((2, 3, 1))(reshaped_input)
    
    # Additional convolution layers to mimic the architecture in the provided MNIST example
    conv1 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(shuffled_input)
    pool1 = MaxPooling2D(pool_size=(2, 2), strides=2, padding='same')(conv1)
    
    conv2 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(pool1)
    pool2 = MaxPooling2D(pool_size=(2, 2), strides=2, padding='same')(conv2)
    
    # Concatenate the outputs of parallel paths
    def concatenate_paths(input_tensor):
        path1 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        path2 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        return Concatenate()(path1, path2)
    
    paths_output = concatenate_paths(input_tensor=conv2)
    paths_output = concatenate_paths(input_tensor=pool2)
    
    # Concatenate paths output with the original pool1
    concatenated_output = Concatenate()([paths_output, pool1])
    
    # Batch normalization and flattening
    bn = BatchNormalization()(concatenated_output)
    flat = Flatten()(bn)
    
    # Fully connected layers
    dense1 = Dense(units=256, activation='relu')(flat)
    dense2 = Dense(units=128, activation='relu')(dense1)
    output_layer = Dense(units=10, activation='softmax')(dense2)
    
    # Construct the model
    model = keras.Model(inputs=input_layer, outputs=output_layer)
    
    return model

# Create and compile the model
model = dl_model()
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])