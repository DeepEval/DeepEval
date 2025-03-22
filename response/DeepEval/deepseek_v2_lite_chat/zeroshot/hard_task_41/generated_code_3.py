from keras.models import Model
from keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dropout, Concatenate, Dense, Reshape, AveragePooling2D

def dl_model():
    # Input layer
    inputs = Input(shape=(28, 28, 1))  # Assuming input images are 28x28 pixels

    # Block 1
    block1_output = inputs
    for i in range(3):
        block1_output = AveragePooling2D(pool_size=(1, 1), strides=(1, 1), padding='same')(block1_output)
        block1_output = Conv2D(filters=64, kernel_size=(2, 2), activation='relu')(block1_output)

    # Flattening and regularization
    block1_output = Flatten()(block1_output)
    block1_output = Dropout(0.5)(block1_output)

    # Concatenate paths from Block 1
    block1_output = Concatenate()([block1_output, block1_output, block1_output])

    # Block 2
    block2_input = block1_output
    for _ in range(4):
        block2_input = Conv2D(filters=64, kernel_size=(1, 1), activation='relu')(block2_input)
        block2_input = Conv2D(filters=64, kernel_size=(3, 3), activation='relu')(block2_input)
        block2_input = Conv2D(filters=64, kernel_size=(3, 3), activation='relu')(block2_input)
        block2_input = AveragePooling2D(pool_size=(1, 1), strides=(1, 1), padding='same')(block2_input)

    # Concatenate branches from Block 2
    block2_output = Concatenate()([block2_input, block2_input, block2_input, block2_output])

    # Fully connected layers and reshaping
    block2_output = Flatten()(block2_output)
    block2_output = Dense(128, activation='relu')(block2_output)
    block2_output = Dense(64, activation='relu')(block2_output)
    output = Dense(10, activation='softmax')(block2_output)  # Assuming 10 classes for MNIST

    # Model
    model = Model(inputs=inputs, outputs=output)

    return model

# Create the model
model = dl_model()
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])