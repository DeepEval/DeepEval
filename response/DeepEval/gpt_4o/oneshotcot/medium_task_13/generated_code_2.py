import keras
from keras.layers import Input, Conv2D, Concatenate, Flatten, Dense

def dl_model():
    input_layer = Input(shape=(32, 32, 3))
    
    # First convolutional layer
    conv1 = Conv2D(filters=32, kernel_size=(3, 3), activation='relu', padding='same')(input_layer)
    concat1 = Concatenate()([input_layer, conv1])  # Concatenate input with conv1 output
    
    # Second convolutional layer
    conv2 = Conv2D(filters=64, kernel_size=(3, 3), activation='relu', padding='same')(concat1)
    concat2 = Concatenate()([concat1, conv2])  # Concatenate concat1 with conv2 output
    
    # Third convolutional layer
    conv3 = Conv2D(filters=128, kernel_size=(3, 3), activation='relu', padding='same')(concat2)
    
    # Flatten the final output
    flatten_layer = Flatten()(conv3)
    
    # Fully connected layers
    dense1 = Dense(units=128, activation='relu')(flatten_layer)
    output_layer = Dense(units=10, activation='softmax')(dense1)  # 10 classes for CIFAR-10
    
    # Construct the model
    model = keras.Model(inputs=input_layer, outputs=output_layer)
    
    return model