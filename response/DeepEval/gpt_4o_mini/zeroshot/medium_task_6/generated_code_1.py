import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, BatchNormalization, ReLU, Flatten, Dense, Add
from tensorflow.keras.models import Model

def dl_model():
    # Input layer
    input_layer = Input(shape=(32, 32, 3))

    # Initial convolution layer
    x_initial = Conv2D(32, (3, 3), padding='same')(input_layer)
    
    # First parallel block
    x1 = Conv2D(32, (3, 3), padding='same')(x_initial)
    x1 = BatchNormalization()(x1)
    x1 = ReLU()(x1)

    # Second parallel block
    x2 = Conv2D(32, (3, 3), padding='same')(x_initial)
    x2 = BatchNormalization()(x2)
    x2 = ReLU()(x2)

    # Third parallel block
    x3 = Conv2D(32, (3, 3), padding='same')(x_initial)
    x3 = BatchNormalization()(x3)
    x3 = ReLU()(x3)

    # Adding the outputs of the parallel blocks to the initial convolution output
    x = Add()([x_initial, x1, x2, x3])
    
    # Flattening the output
    x = Flatten()(x)
    
    # Fully connected layers
    x = Dense(128, activation='relu')(x)
    x = Dense(64, activation='relu')(x)

    # Output layer
    output_layer = Dense(10, activation='softmax')(x)  # 10 classes for CIFAR-10

    # Construct the model
    model = Model(inputs=input_layer, outputs=output_layer)

    return model

# Example usage:
model = dl_model()
model.summary()