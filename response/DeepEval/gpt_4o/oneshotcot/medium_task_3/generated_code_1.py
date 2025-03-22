import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Add, Flatten, Dense

def dl_model():
    # Define the input layer with shape (28, 28, 1) for the MNIST dataset
    input_layer = Input(shape=(28, 28, 1))
    
    # First block: Convolutional Layer followed by MaxPooling
    conv1 = Conv2D(filters=32, kernel_size=(3, 3), padding='same', activation='relu')(input_layer)
    max_pooling1 = MaxPooling2D(pool_size=(2, 2), strides=2, padding='same')(conv1)
    
    # Second block: Another Convolutional Layer and MaxPooling
    conv2 = Conv2D(filters=1, kernel_size=(3, 3), padding='same', activation='relu')(max_pooling1)
    max_pooling2 = MaxPooling2D(pool_size=(2, 2), strides=2, padding='same')(conv2)
    
    # Resize input to match the dimension of the second block's output for addition
    input_resized = MaxPooling2D(pool_size=(4, 4), strides=4, padding='same')(input_layer)
    
    # Combine the output of the second block with the resized input using addition
    added_output = Add()([max_pooling2, input_resized])
    
    # Flatten the combined output
    flatten_layer = Flatten()(added_output)
    
    # Add a fully connected layer for classification
    output_layer = Dense(units=10, activation='softmax')(flatten_layer)
    
    # Create the model
    model = keras.Model(inputs=input_layer, outputs=output_layer)
    
    return model