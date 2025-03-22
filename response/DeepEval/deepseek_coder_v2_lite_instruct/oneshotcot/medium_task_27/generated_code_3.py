import keras
from keras.layers import Input, Conv2D, Add, GlobalAveragePooling2D, Dense, Multiply
from keras.models import Model

def dl_model():
    input_layer = Input(shape=(32, 32, 3))
    
    # First convolutional branch
    conv3x3 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_layer)
    conv3x3 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(conv3x3)
    
    # Second convolutional branch
    conv5x5 = Conv2D(filters=32, kernel_size=(5, 5), strides=(1, 1), padding='same', activation='relu')(input_layer)
    conv5x5 = Conv2D(filters=32, kernel_size=(5, 5), strides=(1, 1), padding='same', activation='relu')(conv5x5)
    
    # Add the outputs of the two branches
    added = Add()([conv3x3, conv5x5])
    
    # Global Average Pooling
    gap = GlobalAveragePooling2D()(added)
    
    # Fully connected layers
    dense1 = Dense(units=128, activation='relu')(gap)
    dense2 = Dense(units=64, activation='relu')(dense1)
    
    # Attention mechanism
    attention_weights = Dense(units=2, activation='softmax')(dense2)
    
    # Multiply attention weights with the outputs of the branches
    attention_branch1 = Multiply()([attention_weights[:, 0:1], conv3x3])
    attention_branch2 = Multiply()([attention_weights[:, 1:2], conv5x5])
    
    # Add the weighted outputs
    weighted_sum = Add()([attention_branch1, attention_branch2])
    
    # Final fully connected layer
    output_layer = Dense(units=10, activation='softmax')(weighted_sum)
    
    model = Model(inputs=input_layer, outputs=output_layer)
    
    return model