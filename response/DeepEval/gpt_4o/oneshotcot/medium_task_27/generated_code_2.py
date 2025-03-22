from keras.layers import Input, Conv2D, Add, GlobalAveragePooling2D, Dense, Multiply
from keras.models import Model

def dl_model():
    input_layer = Input(shape=(32, 32, 3))

    # Branch 1: 3x3 Convolution
    branch1 = Conv2D(filters=64, kernel_size=(3, 3), padding='same', activation='relu')(input_layer)

    # Branch 2: 5x5 Convolution
    branch2 = Conv2D(filters=64, kernel_size=(5, 5), padding='same', activation='relu')(input_layer)

    # Combine branches using addition
    combined = Add()([branch1, branch2])

    # Global Average Pooling
    gap = GlobalAveragePooling2D()(combined)

    # Attention mechanism
    attention_weights1 = Dense(units=64, activation='softmax')(gap)
    attention_weights2 = Dense(units=64, activation='softmax')(gap)

    # Apply attention weights to branches
    weighted_branch1 = Multiply()([branch1, attention_weights1])
    weighted_branch2 = Multiply()([branch2, attention_weights2])

    # Final combination of weighted branches
    final_output = Add()([weighted_branch1, weighted_branch2])

    # Global Average Pooling after attention mechanism
    gap_final = GlobalAveragePooling2D()(final_output)

    # Output fully connected layer
    output_layer = Dense(units=10, activation='softmax')(gap_final)

    # Build the model
    model = Model(inputs=input_layer, outputs=output_layer)

    return model