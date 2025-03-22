import keras
from keras.layers import Input, Conv2D, MaxPooling2D, ReLU, SeparableConv2D, Add, Flatten, Dense
from keras.models import Model

def dl_model():
    # Define the input layer
    input_layer = Input(shape=(28, 28, 1))
    
    # Main path block 1
    conv1 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_layer)
    sep_conv1 = SeparableConv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(conv1)
    max_pool1 = MaxPooling2D(pool_size=(2, 2), strides=2, padding='valid')(sep_conv1)
    
    # Main path block 2
    conv2 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(max_pool1)
    sep_conv2 = SeparableConv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(conv2)
    max_pool2 = MaxPooling2D(pool_size=(2, 2), strides=2, padding='valid')(sep_conv2)
    
    # Branch path
    branch_conv = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_layer)
    
    # Sum the outputs from both paths
    added = Add()([max_pool2, branch_conv])
    
    # Flatten the result
    flatten_layer = Flatten()(added)
    
    # Fully connected layer
    dense1 = Dense(units=128, activation='relu')(flatten_layer)
    output_layer = Dense(units=10, activation='softmax')(dense1)
    
    # Construct the model
    model = Model(inputs=input_layer, outputs=output_layer)
    
    return model