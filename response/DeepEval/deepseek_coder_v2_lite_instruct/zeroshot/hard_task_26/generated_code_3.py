import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, Add, Flatten, Dense

def dl_model():
    # Define the input layer
    inputs = Input(shape=(32, 32, 3))

    # Main path
    # Initial 1x1 convolutional layer
    x = Conv2D(32, (1, 1), activation='relu')(inputs)

    # Branch 1: 3x3 convolutional layer
    branch1 = Conv2D(32, (3, 3), activation='relu')(x)

    # Branch 2: Max pooling followed by 3x3 convolutional layer
    branch2 = MaxPooling2D((2, 2))(x)
    branch2 = Conv2D(32, (3, 3), activation='relu')(branch2)
    branch2 = UpSampling2D((2, 2))(branch2)

    # Branch 3: Max pooling followed by 3x3 convolutional layer
    branch3 = MaxPooling2D((2, 2))(x)
    branch3 = Conv2D(32, (3, 3), activation='relu')(branch3)
    branch3 = UpSampling2D((2, 2))(branch3)

    # Concatenate the outputs from all branches
    merged = tf.keras.layers.concatenate([branch1, branch2, branch3])

    # Final 1x1 convolutional layer
    main_output = Conv2D(64, (1, 1), activation='relu')(merged)

    # Branch path
    # Initial 1x1 convolutional layer
    branch_path = Conv2D(64, (1, 1), activation='relu')(inputs)

    # Add the main path and branch path outputs
    added = Add()([main_output, branch_path])

    # Flatten the output
    flattened = Flatten()(added)

    # Two fully connected layers for classification
    fc1 = Dense(128, activation='relu')(flattened)
    outputs = Dense(10, activation='softmax')(fc1)

    # Define the model
    model = Model(inputs=inputs, outputs=outputs)

    return model

# Example usage:
# model = dl_model()
# model.summary()