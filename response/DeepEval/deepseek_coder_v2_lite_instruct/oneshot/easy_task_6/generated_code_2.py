import keras
from keras.layers import Input, Conv2D, Add, Flatten, Dense

def dl_model():
    # Define the input layer
    input_layer = Input(shape=(28, 28, 1))
    
    # Main path
    conv1 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_layer)
    conv2 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(conv1)
    
    # Branch path
    branch = input_layer
    
    # Add the main path and the branch path
    added = Add()([conv2, branch])
    
    # Flatten the result
    flatten_layer = Flatten()(added)
    
    # Fully connected layer
    dense_layer = Dense(units=128, activation='relu')(flatten_layer)
    
    # Output layer
    output_layer = Dense(units=10, activation='softmax')(dense_layer)
    
    # Construct the model
    model = keras.Model(inputs=input_layer, outputs=output_layer)
    
    return model