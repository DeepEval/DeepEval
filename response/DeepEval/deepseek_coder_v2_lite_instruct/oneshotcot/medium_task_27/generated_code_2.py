import keras
from keras.layers import Input, Conv2D, Add, GlobalAveragePooling2D, Dense, Multiply
from keras.models import Model

def dl_model():
    input_layer = Input(shape=(32, 32, 3))
    
    # First convolutional branch
    conv3x3 = Conv2D(filters=32, kernel_size=(3, 3), padding='same', activation='relu')(input_layer)
    conv3x3 = Conv2D(filters=32, kernel_size=(3, 3), padding='same', activation='relu')(conv3x3)
    
    # Second convolutional branch
    conv5x5 = Conv2D(filters=32, kernel_size=(5, 5), padding='same', activation='relu')(input_layer)
    conv5x5 = Conv2D(filters=32, kernel_size=(5, 5), padding='same', activation='relu')(conv5x5)
    
    # Add the outputs of the two branches
    added = Add()([conv3x3, conv5x5])
    
    # Global average pooling
    gap = GlobalAveragePooling2D()(added)
    
    # Fully connected layers
    dense1 = Dense(units=128, activation='relu')(gap)
    dense2 = Dense(units=64, activation='relu')(dense1)
    
    # Output layer
    output_layer = Dense(units=10, activation='softmax')(dense2)
    
    # Create the model
    model = Model(inputs=input_layer, outputs=output_layer)
    
    return model