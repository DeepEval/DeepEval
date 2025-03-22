import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, Concatenate, BatchNormalization

def dl_model():
    # Input layer
    input_layer = Input(shape=(28, 28, 1))
    
    # Block 1: Three convolutional layers followed by max pooling
    def block1(input_tensor):
        conv1 = Conv2D(filters=32, kernel_size=(3, 3), activation='relu', padding='same')(input_tensor)
        maxpool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
        conv2 = Conv2D(filters=64, kernel_size=(3, 3), activation='relu', padding='same')(maxpool1)
        maxpool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
        conv3 = Conv2D(filters=64, kernel_size=(3, 3), activation='relu', padding='same')(maxpool2)
        return conv3
    
    block1_output = block1(input_layer)
    
    # Block 2: Four convolutional layers followed by max pooling
    def block2(input_tensor):
        conv4 = Conv2D(filters=128, kernel_size=(3, 3), activation='relu', padding='same')(input_tensor)
        maxpool3 = MaxPooling2D(pool_size=(2, 2))(conv4)
        conv5 = Conv2D(filters=128, kernel_size=(3, 3), activation='relu', padding='same')(maxpool3)
        maxpool4 = MaxPooling2D(pool_size=(2, 2))(conv5)
        conv6 = Conv2D(filters=128, kernel_size=(3, 3), activation='relu', padding='same')(maxpool4)
        conv7 = Conv2D(filters=128, kernel_size=(3, 3), activation='relu', padding='same')(conv6)
        return Concatenate()([conv7])
    
    block2_output = block2(block1_output)
    
    # Flatten and fully connected layers
    flatten_layer = Flatten()(block2_output)
    dense1 = Dense(units=128, activation='relu')(flatten_layer)
    dense2 = Dense(units=64, activation='relu')(dense1)
    dense3 = Dense(units=32, activation='relu')(dense2)
    output_layer = Dense(units=10, activation='softmax')(dense3)
    
    # Construct the model
    model = keras.Model(inputs=input_layer, outputs=output_layer)
    
    return model