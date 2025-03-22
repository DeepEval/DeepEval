import keras
from keras.layers import Input, Conv2D, Concatenate, Flatten, Dense

def dl_model():
    # Define the input layer with the shape (32, 32, 64)
    input_layer = Input(shape=(32, 32, 64))
    
    # Compress the input channels with a 1x1 convolution
    conv1x1 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='valid', activation='relu')(input_layer)
    
    # Expand features through two parallel convolutional layers
    conv1x1_parallel = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(conv1x1)
    conv3x3_parallel = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(conv1x1)
    
    # Concatenate the outputs of the parallel layers
    concatenated_output = Concatenate()([conv1x1_parallel, conv3x3_parallel])
    
    # Flatten the output feature map into a one-dimensional vector
    flatten_layer = Flatten()(concatenated_output)
    
    # Fully connected layers
    dense1 = Dense(units=128, activation='relu')(flatten_layer)
    output_layer = Dense(units=10, activation='softmax')(dense1)  # Assuming 10 classes for classification
    
    # Construct the model
    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model