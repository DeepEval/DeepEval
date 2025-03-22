from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, GlobalAveragePooling2D, Flatten, Dense, Activation, Add, Multiply
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

def residual_block(input, filters, kernel_size):
    conv1 = Conv2D(filters, (kernel_size, kernel_size), padding='same')(input)
    bn1 = BatchNormalization()(conv1)
    act1 = Activation('relu')(bn1)

    conv2 = Conv2D(filters, (kernel_size, kernel_size), padding='same')(act1)
    bn2 = BatchNormalization()(conv2)

    # Residual connection
    shortcut = Conv2D(filters, (1, 1), padding='same')(input)
    add = Add()([shortcut, bn2])
    act2 = Activation('relu')(add)

    return act2

def deep_residual_model(input_shape, num_classes):
    # Input layer
    inputs = Input(shape=input_shape)

    # Block 1
    x = Conv2D(64, (3, 3), padding='same')(inputs)
    x = Activation('relu')(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)

    # Block 2
    x = residual_block(x, 64, (3, 3))
    x = residual_block(x, 64, (3, 3))

    # Block 3
    x = residual_block(x, 128, (3, 3))
    x = residual_block(x, 128, (3, 3))

    # Block 4
    x = residual_block(x, 256, (3, 3))
    x = residual_block(x, 256, (3, 3))

    # Block 5
    x = residual_block(x, 512, (3, 3))
    x = residual_block(x, 512, (3, 3))

    # Global average pooling
    avg_pool = GlobalAveragePooling2D()(x)

    # Fully connected layers
    fc1 = Dense(256, activation='relu')(avg_pool)
    fc2 = Dense(num_classes, activation='softmax')(fc1)

    # Create model
    model = Model(inputs=inputs, outputs=fc2)

    return model

# Example usage:
input_shape = (32, 32, 3)
num_classes = 10

model = deep_residual_model(input_shape, num_classes)