import tensorflow as tf
from tensorflow.keras.layers import Input, GlobalAveragePooling2D, Dense, Reshape, Multiply, Flatten
from tensorflow.keras.models import Model

def dl_model():
    # Input layer
    input_shape = (32, 32, 3)  # CIFAR-10 images are 32x32 with 3 channels
    inputs = Input(shape=input_shape)

    # Global Average Pooling to compress the input features
    x = GlobalAveragePooling2D()(inputs)

    # Two fully connected layers to learn channel-wise correlations
    # Assuming we want to generate weights with the same size as the number of channels (3)
    x = Dense(32, activation='relu')(x)
    x = Dense(3, activation='sigmoid')(x)  # Assuming the output size to match the number of channels

    # Reshape weights to align with the input shape (channel-wise)
    x = Reshape((1, 1, 3))(x)

    # Multiply element-wise with the input feature map
    x = Multiply()([inputs, x])

    # Flatten the result
    x = Flatten()(x)

    # Fully connected layer to obtain the final probability distribution
    outputs = Dense(10, activation='softmax')(x)  # CIFAR-10 has 10 classes

    # Construct and return the model
    model = Model(inputs=inputs, outputs=outputs)
    return model

# Example usage
model = dl_model()
model.summary()