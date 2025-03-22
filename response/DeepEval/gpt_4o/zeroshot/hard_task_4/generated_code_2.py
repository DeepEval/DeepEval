import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, DepthwiseConv2D, Dense, GlobalAveragePooling2D, Reshape, Multiply, Flatten
from tensorflow.keras.models import Model

def dl_model():
    input_shape = (32, 32, 3)  # CIFAR-10 input shape
    num_classes = 10  # CIFAR-10 has 10 classes

    # Input layer
    inputs = Input(shape=input_shape)

    # Increase dimensionality of the input's channels threefold with a 1x1 convolution
    x = Conv2D(9, (1, 1), activation='relu')(inputs)  # 3 channels * 3 = 9 channels

    # Extract initial features using a 3x3 depthwise separable convolution
    x = DepthwiseConv2D((3, 3), padding='same', activation='relu')(x)

    # Compute channel attention weights through global average pooling
    gap = GlobalAveragePooling2D()(x)
    fc1 = Dense(9 // 2, activation='relu')(gap)  # Reduce dimensionality in the first FC layer
    fc2 = Dense(9, activation='sigmoid')(fc1)  # Output dimensions must match the channel size

    # Reshape weights to match the initial features
    attention_weights = Reshape((1, 1, 9))(fc2)

    # Multiply with the initial features to achieve channel attention weighting
    x = Multiply()([x, attention_weights])

    # Reduce dimensionality with a 1x1 convolution
    x = Conv2D(3, (1, 1), activation='relu')(x)  # Reduce back to original input channels

    # Combine with the initial input
    combined = tf.keras.layers.add([inputs, x])

    # Flattening layer
    flat = Flatten()(combined)

    # Fully connected layer for classification
    outputs = Dense(num_classes, activation='softmax')(flat)

    # Create the model
    model = Model(inputs=inputs, outputs=outputs)

    return model

# Example usage
# model = dl_model()
# model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
# model.summary()