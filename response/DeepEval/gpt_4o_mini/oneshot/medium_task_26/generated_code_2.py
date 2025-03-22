import keras
from keras.layers import Input, Conv2D, Concatenate, Flatten, Dense

def dl_model():     
    # Define the input layer with shape (32, 32, 64)
    input_layer = Input(shape=(32, 32, 64))
    
    # Compress the input channels with a 1x1 convolutional layer
    compressed = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='valid', activation='relu')(input_layer)

    # Expand features through two parallel convolutional layers
    path1 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(compressed)
    path2 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(compressed)

    # Concatenate the outputs of the two paths
    concatenated = Concatenate()([path1, path2])
    
    # Flatten the output feature map into a one-dimensional vector
    flatten_layer = Flatten()(concatenated)
    
    # Fully connected layers
    dense1 = Dense(units=128, activation='relu')(flatten_layer)
    output_layer = Dense(units=10, activation='softmax')(dense1)  # Assuming 10 classes for classification

    # Create the model
    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model