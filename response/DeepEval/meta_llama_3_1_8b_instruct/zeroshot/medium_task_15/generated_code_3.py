from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Model

def dl_model():
    # Define the input layer (32x32 images with 3 color channels)
    inputs = keras.Input(shape=(32, 32, 3), name='inputs')

    # Convolutional layer to extract initial features
    x = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(inputs)
    x = layers.BatchNormalization()(x)

    # Compress feature maps using global average pooling
    x = layers.GlobalAveragePooling2D()(x)

    # Two fully connected layers to adjust dimensions
    x = layers.Dense(128, activation='relu')(x)
    x = layers.Dense(128, activation='relu')(x)

    # Reshape output to match initial feature dimensions
    x = layers.Reshape((1, 1, 128))(x)

    # Multiply weighted feature maps with initial features
    x = layers.Conv2D(32, (1, 1), activation='relu', padding='same')(x)
    x = layers.multiply([x, inputs])

    # Concatenate with input layer
    x = layers.Concatenate()([x, inputs])

    # Reduce dimensionality and downsample using 1x1 convolution and average pooling
    x = layers.Conv2D(16, (1, 1), activation='relu')(x)
    x = layers.AveragePooling2D((2, 2))(x)

    # Output layer for classification
    outputs = layers.Dense(10, activation='softmax')(x)

    # Create the model
    model = Model(inputs=inputs, outputs=outputs)

    return model

# Create and compile the model
model = dl_model()
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])