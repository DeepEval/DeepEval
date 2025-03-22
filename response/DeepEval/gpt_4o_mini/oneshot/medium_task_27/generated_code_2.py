import keras
from keras.layers import Input, Conv2D, GlobalAveragePooling2D, Dense, Add, Multiply
from keras.models import Model

def dl_model():
    input_layer = Input(shape=(32, 32, 3))

    # First convolutional branch with 3x3 kernel
    branch1 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_layer)

    # Second convolutional branch with 5x5 kernel
    branch2 = Conv2D(filters=32, kernel_size=(5, 5), strides=(1, 1), padding='same', activation='relu')(input_layer)

    # Adding the outputs of the two branches
    merged = Add()([branch1, branch2])

    # Global average pooling
    pooled = GlobalAveragePooling2D()(merged)

    # Two fully connected layers
    dense1 = Dense(units=128, activation='relu')(pooled)
    attention_weights = Dense(units=2, activation='softmax')(dense1)  # 2 weights for each branch

    # Multiply branch outputs by their respective attention weights
    weighted_branch1 = Multiply()([branch1, attention_weights[0]])
    weighted_branch2 = Multiply()([branch2, attention_weights[1]])
    
    # Combine the weighted outputs
    combined_output = Add()([weighted_branch1, weighted_branch2])
    
    # Final fully connected layer for class probabilities
    final_output = Dense(units=10, activation='softmax')(combined_output)

    model = Model(inputs=input_layer, outputs=final_output)

    return model