import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, BatchNormalization, ReLU, Add, Flatten, Dense
from tensorflow.keras.models import Model

def dl_model():
    input_shape = (32, 32, 3)  # CIFAR-10 images are 32x32 pixels with 3 color channels
    num_classes = 10           # CIFAR-10 has 10 classes

    inputs = Input(shape=input_shape)

    # Block 1
    x1 = Conv2D(32, (3, 3), padding='same')(inputs)
    x1 = BatchNormalization()(x1)
    x1 = ReLU()(x1)

    # Block 2
    x2 = Conv2D(64, (3, 3), padding='same')(x1)
    x2 = BatchNormalization()(x2)
    x2 = ReLU()(x2)

    # Block 3
    x3 = Conv2D(128, (3, 3), padding='same')(x2)
    x3 = BatchNormalization()(x3)
    x3 = ReLU()(x3)

    # Parallel branch processing the input directly
    parallel_branch = Conv2D(128, (3, 3), padding='same')(inputs)
    parallel_branch = BatchNormalization()(parallel_branch)
    parallel_branch = ReLU()(parallel_branch)

    # Add outputs of block 3 and parallel branch
    combined = Add()([x3, parallel_branch])

    # Fully connected layers for classification
    flat = Flatten()(combined)
    fc1 = Dense(256, activation='relu')(flat)
    fc2 = Dense(num_classes, activation='softmax')(fc1)

    # Create model
    model = Model(inputs=inputs, outputs=fc2)

    return model

# Create the model instance
model = dl_model()

# Print the model summary to verify the architecture
model.summary()