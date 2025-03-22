import keras
from keras.layers import Input, Conv2D, Add, GlobalAveragePooling2D, Dense, Multiply
from keras.models import Model

def dl_model():
    input_layer = Input(shape=(32, 32, 3))
    
    # First parallel convolutional branch (3x3)
    branch1 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_layer)
    branch1 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(branch1)

    # Second parallel convolutional branch (5x5)
    branch2 = Conv2D(filters=32, kernel_size=(5, 5), strides=(1, 1), padding='same', activation='relu')(input_layer)
    branch2 = Conv2D(filters=64, kernel_size=(5, 5), strides=(1, 1), padding='same', activation='relu')(branch2)

    # Combine the outputs of the two branches through addition
    merged = Add()([branch1, branch2])

    # Global average pooling layer
    pooled = GlobalAveragePooling2D()(merged)

    # Fully connected layers
    dense1 = Dense(units=128, activation='relu')(pooled)
    dense2 = Dense(units=64, activation='relu')(dense1)
    
    # Output layer with softmax for class probabilities
    output_layer = Dense(units=10, activation='softmax')(dense2)

    # Create the model
    model = Model(inputs=input_layer, outputs=output_layer)

    return model