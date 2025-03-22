import keras
from keras.layers import Input, Conv2D, Add, Flatten, Dense

def dl_model():
    
    # Input layer for CIFAR-10 images (32x32 RGB images)
    input_layer = Input(shape=(32, 32, 3))
    
    # First sequential convolutional layer
    conv1 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_layer)
    
    # Second sequential convolutional layer
    conv2 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(conv1)
    
    # Third sequential convolutional layer
    conv3 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(conv2)
    
    # Separate convolutional layer processing the input directly
    direct_conv = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_layer)
    
    # Adding the outputs of the convolutional layers
    added_outputs = Add()([conv1, conv2, conv3, direct_conv])
    
    # Flatten the result
    flatten_layer = Flatten()(added_outputs)
    
    # Fully connected layers
    dense1 = Dense(units=128, activation='relu')(flatten_layer)
    output_layer = Dense(units=10, activation='softmax')(dense1)
    
    # Construct the model
    model = keras.Model(inputs=input_layer, outputs=output_layer)
    
    return model