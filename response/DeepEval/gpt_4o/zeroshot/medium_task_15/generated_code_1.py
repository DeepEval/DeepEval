from tensorflow.keras.layers import Input, Conv2D, BatchNormalization, ReLU, GlobalAveragePooling2D, Dense, Multiply, Concatenate, AveragePooling2D, Flatten
from tensorflow.keras.models import Model

def dl_model():
    # Input layer for CIFAR-10 images (32x32x3)
    input_layer = Input(shape=(32, 32, 3))

    # Initial convolutional layer
    conv1 = Conv2D(32, (3, 3), padding='same')(input_layer)
    bn1 = BatchNormalization()(conv1)
    relu1 = ReLU()(bn1)

    # Global Average Pooling
    gap = GlobalAveragePooling2D()(relu1)

    # Fully connected layers to adjust dimensions to the initial features' channels
    dense1 = Dense(32, activation='relu')(gap)
    dense2 = Dense(32, activation='sigmoid')(dense1)

    # Reshape output to match the initial feature map size
    reshaped_dense = tf.keras.layers.Reshape((1, 1, 32))(dense2)

    # Element-wise multiplication
    weighted_features = Multiply()([relu1, reshaped_dense])

    # Concatenate weighted features with input
    concatenated = Concatenate()([input_layer, weighted_features])

    # Reduce dimensionality and downsample
    conv2 = Conv2D(32, (1, 1), activation='relu')(concatenated)
    pooled = AveragePooling2D((2, 2))(conv2)

    # Flatten and add final fully connected layer for classification
    flat = Flatten()(pooled)
    output = Dense(10, activation='softmax')(flat)

    # Construct and return the model
    model = Model(inputs=input_layer, outputs=output)
    return model