import keras
from keras.layers import Input, Conv2D, Dropout, Add, Flatten, Dense

def dl_model():
    # Input layer for 28x28 grayscale images
    input_layer = Input(shape=(28, 28, 1))
    
    # Main path with two convolution and dropout blocks
    conv1 = Conv2D(filters=32, kernel_size=(3, 3), padding='same', activation='relu')(input_layer)
    dropout1 = Dropout(0.25)(conv1)
    
    conv2 = Conv2D(filters=64, kernel_size=(3, 3), padding='same', activation='relu')(dropout1)
    dropout2 = Dropout(0.25)(conv2)
    
    # Another convolutional layer to restore the number of channels (e.g., back to 1 channel)
    conv3 = Conv2D(filters=1, kernel_size=(3, 3), padding='same', activation='relu')(dropout2)

    # Branch path directly connected to the input
    branch_path = input_layer
    
    # Combine main path and branch path using an addition operation
    combined = Add()([conv3, branch_path])
    
    # Flattening layer followed by a fully connected layer
    flatten_layer = Flatten()(combined)
    dense_layer = Dense(units=10, activation='softmax')(flatten_layer)
    
    # Create the model
    model = keras.Model(inputs=input_layer, outputs=dense_layer)
    
    return model