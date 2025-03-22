import keras
from keras.layers import Input, Conv2D, GlobalAveragePooling2D, Dense, Add, Multiply
from keras.models import Model

def dl_model():
    input_layer = Input(shape=(32, 32, 3))

    # First branch with 3x3 convolution
    branch1 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_layer)
    
    # Second branch with 5x5 convolution
    branch2 = Conv2D(filters=32, kernel_size=(5, 5), strides=(1, 1), padding='same', activation='relu')(input_layer)

    # Combine branches using addition
    combined = Add()([branch1, branch2])

    # Global average pooling
    pooled = GlobalAveragePooling2D()(combined)

    # Fully connected layers
    dense1 = Dense(units=128, activation='relu')(pooled)
    dense2 = Dense(units=64, activation='relu')(dense1)

    # Attention weights for each branch
    attention_weights = Dense(units=2, activation='softmax')(dense2)

    # Split the attention weights
    branch1_weight = attention_weights[:, 0]
    branch2_weight = attention_weights[:, 1]

    # Multiply each branch output by its corresponding attention weight
    weighted_branch1 = Multiply()([branch1, branch1_weight])
    weighted_branch2 = Multiply()([branch2, branch2_weight])

    # Combine the weighted outputs
    final_output = Add()([weighted_branch1, weighted_branch2])

    # Fully connected layer for final classification
    output_layer = Dense(units=10, activation='softmax')(final_output)

    # Create the model
    model = Model(inputs=input_layer, outputs=output_layer)

    return model