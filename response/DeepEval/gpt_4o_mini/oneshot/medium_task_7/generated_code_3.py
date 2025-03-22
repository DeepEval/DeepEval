import keras
from keras.layers import Input, Conv2D, Add, Flatten, Dense

def dl_model():
    input_layer = Input(shape=(32, 32, 3))  # CIFAR-10 images are 32x32 pixels with 3 color channels (RGB)

    # First convolutional layer
    conv1 = Conv2D(filters=32, kernel_size=(3, 3), padding='same', activation='relu')(input_layer)
    
    # Second convolutional layer
    conv2 = Conv2D(filters=64, kernel_size=(3, 3), padding='same', activation='relu')(conv1)
    
    # Third convolutional layer
    conv3 = Conv2D(filters=128, kernel_size=(3, 3), padding='same', activation='relu')(conv2)
    
    # Direct convolutional path from input
    direct_conv = Conv2D(filters=128, kernel_size=(1, 1), padding='same', activation='relu')(input_layer)
    
    # Adding the outputs from the first two convolutions and the third convolution
    added_output = Add()([conv1, conv2, conv3])  # Adding outputs from first two convolutions and the third one
    combined_output = Add()([added_output, direct_conv])  # Adding the direct path output

    # Flatten the combined output
    flatten_layer = Flatten()(combined_output)
    
    # Fully connected layers
    dense1 = Dense(units=256, activation='relu')(flatten_layer)
    dense2 = Dense(units=128, activation='relu')(dense1)
    
    # Output layer
    output_layer = Dense(units=10, activation='softmax')(dense2)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model