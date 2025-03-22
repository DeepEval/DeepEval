import keras
from keras.layers import Input, Conv2D, Add, Flatten, Dense

def dl_model():
    input_layer = Input(shape=(32, 32, 3))
    
    # First convolutional layer
    conv1 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_layer)
    
    # Second convolutional layer
    conv2 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(conv1)
    
    # Third convolutional layer
    conv3 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(conv2)
    
    # Adding the outputs of the first two layers with the third
    added_conv = Add()([conv1, conv2, conv3])
    
    # A separate convolutional layer processing the input directly
    direct_conv = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_layer)
    
    # Adding the outputs from all paths
    final_add = Add()([added_conv, direct_conv])
    
    # Flattening the final output
    flatten_layer = Flatten()(final_add)
    
    # Fully connected layers for classification
    dense1 = Dense(units=128, activation='relu')(flatten_layer)
    output_layer = Dense(units=10, activation='softmax')(dense1)
    
    # Creating the model
    model = keras.Model(inputs=input_layer, outputs=output_layer)
    
    return model