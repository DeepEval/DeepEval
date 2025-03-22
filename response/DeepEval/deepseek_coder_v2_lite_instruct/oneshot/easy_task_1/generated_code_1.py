import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense

def dl_model():
    # Input layer
    input_layer = Input(shape=(28, 28, 1))
    
    # First convolutional layer
    conv1 = Conv2D(filters=32, kernel_size=(3, 3), activation='relu')(input_layer)
    max_pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
    
    # Second convolutional layer
    conv2 = Conv2D(filters=64, kernel_size=(3, 3), activation='relu')(max_pool1)
    max_pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
    
    # Flatten the output
    flatten_layer = Flatten()(max_pool2)
    
    # First fully connected layer
    dense1 = Dense(units=128, activation='relu')(flatten_layer)
    
    # Second fully connected layer
    output_layer = Dense(units=10, activation='softmax')(dense1)
    
    # Create the model
    model = keras.Model(inputs=input_layer, outputs=output_layer)
    
    return model