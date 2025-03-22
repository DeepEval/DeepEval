import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, GlobalAveragePooling2D, Dense, Concatenate, Reshape, Multiply
from tensorflow.keras.models import Model

def dl_model():
    # Step 1: Input layer
    input_layer = Input(shape=(32, 32, 3))  # CIFAR-10 images are 32x32 with 3 channels (RGB)

    # Step 2: Create the four parallel branches
    branch1 = Conv2D(32, (1, 1), padding='same', activation='relu')(input_layer)  # 1x1 Conv
    branch2 = Conv2D(32, (3, 3), padding='same', activation='relu')(input_layer)  # 3x3 Conv
    branch3 = Conv2D(32, (5, 5), padding='same', activation='relu')(input_layer)  # 5x5 Conv
    branch4 = MaxPooling2D(pool_size=(3, 3), strides=1, padding='same')(input_layer)  # 3x3 Max Pooling

    # Step 3: Concatenate the outputs of the branches
    concatenated = Concatenate()([branch1, branch2, branch3, branch4])

    # Step 4: Global average pooling
    pooled = GlobalAveragePooling2D()(concatenated)

    # Step 5: Two fully connected layers to generate weights
    dense1 = Dense(32, activation='relu')(pooled)  # First fully connected layer
    dense2 = Dense(concatenated.shape[-1], activation='sigmoid')(dense1)  # Second fully connected layer

    # Step 6: Reshape and multiply element-wise
    reshaped_weights = Reshape((1, 1, concatenated.shape[-1]))(dense2)  # Reshape to match input shape for multiplication
    multiplied = Multiply()([concatenated, reshaped_weights])  # Element-wise multiplication

    # Step 7: Final fully connected layer for output
    output = Dense(10, activation='softmax')(GlobalAveragePooling2D()(multiplied))  # 10 classes for CIFAR-10

    # Construct the model
    model = Model(inputs=input_layer, outputs=output)

    return model

# Example of how to create the model
model = dl_model()
model.summary()