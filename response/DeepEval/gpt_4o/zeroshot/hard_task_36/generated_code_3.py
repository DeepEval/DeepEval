import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Dropout, Add, GlobalAveragePooling2D, Flatten, Dense
from tensorflow.keras.models import Model

def dl_model():
    # Input layer
    input_layer = Input(shape=(28, 28, 1))  # MNIST images are 28x28 pixels with 1 channel (grayscale)

    # Main pathway
    # 3x3 Convolution
    x = Conv2D(32, (3, 3), activation='relu', padding='same')(input_layer)
    # Two 1x1 Convolutions
    x = Conv2D(32, (1, 1), activation='relu', padding='same')(x)
    x = Conv2D(32, (1, 1), activation='relu', padding='same')(x)
    # Max Pooling
    x = MaxPooling2D(pool_size=(2, 2))(x)
    # Dropout
    x = Dropout(0.5)(x)

    # Branch pathway
    # Convolution to match the size of the main pathway output
    y = Conv2D(32, (3, 3), activation='relu', padding='same')(input_layer)
    y = MaxPooling2D(pool_size=(2, 2))(y)

    # Fusion of pathways
    fused = Add()([x, y])

    # Global Average Pooling
    global_avg_pool = GlobalAveragePooling2D()(fused)

    # Flatten
    flatten = Flatten()(global_avg_pool)

    # Fully connected layer for classification (10 classes for MNIST)
    output = Dense(10, activation='softmax')(flatten)

    # Create model
    model = Model(inputs=input_layer, outputs=output)

    # Compile the model
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    return model

# Example of creating the model
model = dl_model()
model.summary()