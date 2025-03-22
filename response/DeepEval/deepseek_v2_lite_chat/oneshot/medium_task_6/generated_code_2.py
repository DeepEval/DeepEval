import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, BatchNormalization, Flatten, Dense, ReLU
from keras.models import Model

def dl_model():
    # Input layer
    input_layer = Input(shape=(32, 32, 3))
    
    # Initial convolution
    conv1 = Conv2D(filters=32, kernel_size=(3, 3), padding='same')(input_layer)
    
    # First block of three parallel paths
    def block1():
        conv1 = Conv2D(filters=64, kernel_size=(3, 3), padding='same')(conv1)
        bn1 = BatchNormalization()(conv1)
        relu1 = ReLU()(bn1)
        return relu1
    
    block1_output = block1()
    
    def block2():
        conv1 = Conv2D(filters=64, kernel_size=(3, 3), padding='same')(conv1)
        bn1 = BatchNormalization()(conv1)
        relu1 = ReLU()(bn1)
        return relu1
    
    block2_output = block2()
    
    def block3():
        conv1 = Conv2D(filters=64, kernel_size=(3, 3), padding='same')(conv1)
        bn1 = BatchNormalization()(conv1)
        relu1 = ReLU()(bn1)
        return relu1
    
    block3_output = block3()
    
    # Concatenate outputs of parallel blocks
    concatenated = Concatenate(axis=-1)([block1_output, block2_output, block3_output])
    
    # Fully connected layers
    flatten = Flatten()(concatenated)
    dense1 = Dense(units=256, activation='relu')(flatten)
    dense2 = Dense(units=128, activation='relu')(dense1)
    output_layer = Dense(units=10, activation='softmax')(dense2)
    
    # Model construction
    model = Model(inputs=input_layer, outputs=output_layer)
    
    return model

# Instantiate and compile the model
model = dl_model()
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])