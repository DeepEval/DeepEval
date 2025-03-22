import keras
from keras.layers import Input, Conv2D, Concatenate, Flatten, Dense

def dl_model():
    input_layer = Input(shape=(32, 32, 64))
    
    # Compress the input channels with a 1x1 convolutional layer
    conv1 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_layer)
    
    # Expand the features through two parallel convolutional layers, 
    # applying 1x1 and 3x3 convolutions, and concatenate the results
    conv2 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_layer)
    output_tensor = Concatenate()([conv1, conv2])
    
    # Flatten the output feature map into a one-dimensional vector
    flatten_layer = Flatten()(output_tensor)
    
    # Pass the flattened output through two fully connected layers to produce the classification results
    dense1 = Dense(units=128, activation='relu')(flatten_layer)
    output_layer = Dense(units=10, activation='softmax')(dense1)
    
    # Construct the model
    model = keras.Model(inputs=input_layer, outputs=output_layer)
    
    return model