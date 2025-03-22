import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, BatchNormalization, Flatten, Dense, AveragePooling2D

def dl_model():
    # Input layer
    input_layer = Input(shape=(28, 28, 1))
    
    # Block 1
    def block1(input_tensor):
        conv1 = Conv2D(filters=32, kernel_size=(3, 3), padding='same', activation='relu')(input_tensor)
        avg_pool1_1x1 = AveragePooling2D((1, 1))(conv1)
        avg_pool1_2x2 = AveragePooling2D((2, 2), strides=(2, 2))(conv1)
        avg_pool1_4x4 = AveragePooling2D((4, 4), strides=(4, 4))(conv1)
        
        flat1 = Flatten()(Concatenate()([avg_pool1_1x1, avg_pool1_2x2, avg_pool1_4x4]))
        
        dense1 = Dense(units=128, activation='relu')(flat1)
        fully_connected = Dense(units=64)(dense1)
        reshape1 = Reshape((4, 4, 1))(fully_connected)  # Reshape for Block 2
        
        return reshape1
    
    block1_output = block1(input_tensor=input_layer)
    
    # Block 2
    def block2(input_tensor):
        conv2_1x1 = Conv2D(filters=64, kernel_size=(1, 1), padding='same', activation='relu')(input_tensor)
        conv2_3x3 = Conv2D(filters=64, kernel_size=(3, 3), padding='same', activation='relu')(input_tensor)
        conv2_5x5 = Conv2D(filters=64, kernel_size=(5, 5), padding='same', activation='relu')(input_tensor)
        avg_pool2 = AveragePooling2D((1, 1))(conv2_5x5)

        conv2_1x7_7x1 = Conv2D(filters=64, kernel_size=(1, 7), padding='same')(conv2_3x3)
        conv2_7x1_1x7 = Conv2D(filters=64, kernel_size=(7, 1), padding='same')(conv2_1x7_7x1)
        conv2_3x3_avgpool = AveragePooling2D((3, 3), strides=(1, 1))(conv2_1x7_7x1)
        concat2 = Concatenate()([conv2_1x1, conv2_3x3, conv2_5x5, conv2_3x3_avgpool])

        batch_norm2 = BatchNormalization()(concat2)
        flatten2 = Flatten()(batch_norm2)
        dense2 = Dense(units=128, activation='relu')(flatten2)
        fully_connected2 = Dense(units=64)(dense2)
        reshape2 = Reshape((16, 2, 2))(fully_connected2)  # Reshape for output layer
        
        return reshape2
    
    block2_output = block2(block1_output)
    
    # Output layer
    output_layer = Dense(units=10, activation='softmax')(block2_output)
    
    # Model construction
    model = keras.Model(inputs=input_layer, outputs=output_layer)
    
    return model

# Create the model
model = dl_model()
model.summary()