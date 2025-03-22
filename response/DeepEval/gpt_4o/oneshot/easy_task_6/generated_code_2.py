import keras
from keras.layers import Input, Conv2D, Add, Flatten, Dense

def dl_model():
    input_layer = Input(shape=(28, 28, 1))
    
    # Main path
    # First convolutional layer increases the feature map width
    conv1 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_layer)
    
    # Second convolutional layer restores the number of channels
    conv2 = Conv2D(filters=1, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(conv1)
    
    # Branch path - directly connected to the input
    branch = Conv2D(filters=1, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_layer)
    
    # Combine the main and branch paths through an addition operation
    added = Add()([conv2, branch])
    
    # Flatten the combined output
    flatten_layer = Flatten()(added)
    
    # Fully connected layer to produce final classification
    output_layer = Dense(units=10, activation='softmax')(flatten_layer)
    
    # Create the model
    model = keras.Model(inputs=input_layer, outputs=output_layer)
    
    return model