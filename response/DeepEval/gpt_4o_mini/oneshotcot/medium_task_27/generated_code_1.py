import keras
from keras.layers import Input, Conv2D, Add, GlobalAveragePooling2D, Dense, Multiply
from keras.models import Model

def dl_model():
    
    # Input layer
    input_layer = Input(shape=(32, 32, 3))
    
    # First branch with 3x3 convolution
    branch1 = Conv2D(filters=32, kernel_size=(3, 3), padding='same', activation='relu')(input_layer)
    branch1 = Conv2D(filters=32, kernel_size=(3, 3), padding='same', activation='relu')(branch1)
    
    # Second branch with 5x5 convolution
    branch2 = Conv2D(filters=32, kernel_size=(5, 5), padding='same', activation='relu')(input_layer)
    branch2 = Conv2D(filters=32, kernel_size=(5, 5), padding='same', activation='relu')(branch2)
    
    # Combine branches through addition
    combined = Add()([branch1, branch2])
    
    # Global Average Pooling layer
    gap = GlobalAveragePooling2D()(combined)
    
    # Fully connected layers
    dense1 = Dense(units=128, activation='relu')(gap)
    dense2 = Dense(units=64, activation='relu')(dense1)
    
    # Attention weights
    attention_weights = Dense(units=10, activation='softmax')(dense2)
    
    # Final output (weighted output)
    weighted_output = Multiply()([gap, attention_weights])
    
    # Final classification layer
    final_output = Dense(units=10, activation='softmax')(weighted_output)
    
    # Constructing the model
    model = Model(inputs=input_layer, outputs=final_output)

    return model