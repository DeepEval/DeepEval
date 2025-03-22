import keras
from keras.layers import Input, Conv2D, Concatenate, Add, GlobalAveragePooling2D, Dense

def dl_model():
    
    input_layer = Input(shape=(32, 32, 3))
    
    # Define the path that adds the outputs of the first two convolutional layers with the output of the third convolutional layer
    conv1 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_layer)
    conv2 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(conv1)
    conv3 = Conv2D(filters=128, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(conv2)
    add_output1 = Add()([conv1, conv2, conv3])
    
    # Define the path that directly processes the input
    conv4 = Conv2D(filters=128, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_layer)
    
    # Add the outputs from both paths
    add_output = Add()([add_output1, conv4])
    
    # Use global average pooling to reduce the spatial dimensions of the output
    pool_output = GlobalAveragePooling2D()(add_output)
    
    # Define the two fully connected layers for classification
    dense1 = Dense(units=128, activation='relu')(pool_output)
    output_layer = Dense(units=10, activation='softmax')(dense1)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model