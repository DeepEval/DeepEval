import keras
from keras.layers import Input, Conv2D, Add, Flatten, Dense

def dl_model():
    # CIFAR-10 images are 32x32 with 3 color channels
    input_layer = Input(shape=(32, 32, 3))
    
    # First sequence of convolutional layers
    conv1 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_layer)
    conv2 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(conv1)
    conv3 = Conv2D(filters=128, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(conv2)
    
    # Separate convolutional path directly from input
    separate_conv = Conv2D(filters=128, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_layer)
    
    # Add the outputs from the sequential and separate convolutional paths
    added_outputs = Add()([conv3, separate_conv])
    
    # Flattening and fully connected layers for classification
    flatten_layer = Flatten()(added_outputs)
    dense1 = Dense(units=256, activation='relu')(flatten_layer)
    output_layer = Dense(units=10, activation='softmax')(dense1)
    
    # Construct the model
    model = keras.Model(inputs=input_layer, outputs=output_layer)
    
    return model