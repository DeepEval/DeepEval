import keras
from keras.layers import Input, Conv2D, BatchNormalization, ReLU, Add, Flatten, Dense
from keras.datasets import cifar10

def dl_model():
    input_layer = Input(shape=(32, 32, 3))
    
    # Initial convolutional layer
    initial_conv = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_layer)
    
    # Block 1
    conv1 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same')(initial_conv)
    bn1 = BatchNormalization()(conv1)
    relu1 = ReLU()(bn1)
    
    # Block 2
    conv2 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same')(initial_conv)
    bn2 = BatchNormalization()(conv2)
    relu2 = ReLU()(bn2)
    
    # Block 3
    conv3 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same')(initial_conv)
    bn3 = BatchNormalization()(conv3)
    relu3 = ReLU()(bn3)
    
    # Adding the outputs of the blocks with the initial convolution
    added_output = Add()([initial_conv, relu1, relu2, relu3])
    
    # Flatten and fully connected layers
    flatten_layer = Flatten()(added_output)
    fc1 = Dense(units=128, activation='relu')(flatten_layer)
    fc2 = Dense(units=64, activation='relu')(fc1)
    output_layer = Dense(units=10, activation='softmax')(fc2)
    
    model = keras.Model(inputs=input_layer, outputs=output_layer)
    
    return model