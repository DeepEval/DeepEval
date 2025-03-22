import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, Lambda, Add, Flatten, Dense
from tensorflow.keras.models import Model

def dl_model():
    # Input layer for CIFAR-10 images (32x32 grayscale images)
    inputs = Input(shape=(32, 32, 3))

    # Split the input into three groups along the channel dimension
    splits = Lambda(lambda x: tf.split(x, num_or_size_splits=3, axis=-1))(inputs)

    # Define a function to create the convolutional path for each split
    def create_conv_path(input_tensor):
        x = Conv2D(32, (1, 1), padding='same', activation='relu')(input_tensor)
        x = Conv2D(64, (3, 3), padding='same', activation='relu')(x)
        x = Conv2D(32, (1, 1), padding='same', activation='relu')(x)
        return x

    # Apply the convolutional path to each split
    conv_outputs = [create_conv_path(split) for split in splits]

    # Combine the outputs from the three paths using addition
    main_path = Add()(conv_outputs)

    # Fuse the main path with the original input
    fused = Add()([main_path, inputs])

    # Flatten the combined features
    flattened = Flatten()(fused)

    # Fully connected layer for classification
    outputs = Dense(10, activation='softmax')(flattened)

    # Create the model
    model = Model(inputs=inputs, outputs=outputs)

    return model

# Example usage
model = dl_model()
model.summary()