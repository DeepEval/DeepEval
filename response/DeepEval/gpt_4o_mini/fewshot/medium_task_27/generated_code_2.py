import keras
from keras.layers import Input, Conv2D, Add, GlobalAveragePooling2D, Dense, Multiply
from keras.models import Model

def dl_model():
    input_layer = Input(shape=(32, 32, 3))  # CIFAR-10 images are 32x32 with 3 color channels

    # Branch 1: 3x3 Convolution
    branch1 = Conv2D(filters=32, kernel_size=(3, 3), padding='same', activation='relu')(input_layer)

    # Branch 2: 5x5 Convolution
    branch2 = Conv2D(filters=32, kernel_size=(5, 5), padding='same', activation='relu')(input_layer)

    # Combine branches using addition
    combined = Add()([branch1, branch2])

    # Global Average Pooling
    pooled = GlobalAveragePooling2D()(combined)

    # Fully connected layers for generating attention weights
    dense1 = Dense(units=64, activation='relu')(pooled)
    attention_weights = Dense(units=32, activation='sigmoid')(dense1)  # Generate attention weights for branch outputs

    # Split pooled outputs for multiplication with attention weights
    branch1_pooled = GlobalAveragePooling2D()(branch1)
    branch2_pooled = GlobalAveragePooling2D()(branch2)

    # Multiply outputs of each branch by their corresponding attention weights
    weighted_branch1 = Multiply()([branch1_pooled, attention_weights])
    weighted_branch2 = Multiply()([branch2_pooled, attention_weights])

    # Combine weighted outputs
    final_combined = Add()([weighted_branch1, weighted_branch2])

    # Fully connected layer for final classification
    output_layer = Dense(units=10, activation='softmax')(final_combined)  # 10 classes for CIFAR-10

    # Create the model
    model = Model(inputs=input_layer, outputs=output_layer)

    return model