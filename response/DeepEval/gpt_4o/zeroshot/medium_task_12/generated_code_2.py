import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, BatchNormalization, ReLU, Concatenate, Flatten, Dense
from tensorflow.keras.models import Model

def dl_model():
    # Define the input layer
    inputs = Input(shape=(32, 32, 3))  # CIFAR-10 images are 32x32x3

    # Block 1
    x1 = Conv2D(32, (3, 3), padding='same')(inputs)
    x1 = BatchNormalization()(x1)
    x1 = ReLU()(x1)
    
    # Block 2
    x2 = Conv2D(64, (3, 3), padding='same')(x1)
    x2 = BatchNormalization()(x2)
    x2 = ReLU()(x2)
    x2 = Concatenate()([x1, x2])  # Concatenate output of block 1 and block 2

    # Block 3
    x3 = Conv2D(128, (3, 3), padding='same')(x2)
    x3 = BatchNormalization()(x3)
    x3 = ReLU()(x3)
    x3 = Concatenate()([x2, x3])  # Concatenate output of block 2 and block 3

    # Flatten the output
    x = Flatten()(x3)

    # Fully connected layers
    x = Dense(256, activation='relu')(x)
    outputs = Dense(10, activation='softmax')(x)  # CIFAR-10 has 10 classes

    # Define the model
    model = Model(inputs=inputs, outputs=outputs)

    # Compile the model
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    return model