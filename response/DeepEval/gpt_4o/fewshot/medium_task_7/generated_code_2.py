import keras
from keras.layers import Input, Conv2D, Add, Flatten, Dense
from keras.models import Model

def dl_model():
    # Define the input layer with CIFAR-10 image shape
    input_layer = Input(shape=(32, 32, 3))
    
    # First sequential convolutional layer
    conv1 = Conv2D(filters=32, kernel_size=(3, 3), padding='same', activation='relu')(input_layer)
    
    # Second sequential convolutional layer
    conv2 = Conv2D(filters=64, kernel_size=(3, 3), padding='same', activation='relu')(conv1)
    
    # Third sequential convolutional layer
    conv3 = Conv2D(filters=128, kernel_size=(3, 3), padding='same', activation='relu')(conv2)
    
    # Adding outputs from the first two convolutional layers with the third
    added_path = Add()([conv1, conv2, conv3])
    
    # Separate convolutional layer processing the input directly
    direct_conv = Conv2D(filters=128, kernel_size=(3, 3), padding='same', activation='relu')(input_layer)
    
    # Adding the direct path output with the sequential path output
    final_addition = Add()([added_path, direct_conv])
    
    # Flattening the final output
    flatten_layer = Flatten()(final_addition)
    
    # Fully connected layers for classification
    dense1 = Dense(units=256, activation='relu')(flatten_layer)
    output_layer = Dense(units=10, activation='softmax')(dense1)
    
    # Defining the model
    model = Model(inputs=input_layer, outputs=output_layer)

    return model