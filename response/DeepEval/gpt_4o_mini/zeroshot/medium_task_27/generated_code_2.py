import numpy as np
from keras.datasets import cifar10
from keras.models import Model
from keras.layers import Input, Conv2D, GlobalAveragePooling2D, Dense, Add, Multiply, Softmax
from keras.utils import to_categorical

def dl_model():
    # Input Layer
    inputs = Input(shape=(32, 32, 3))

    # First Convolutional Branch (3x3 Convolutions)
    branch1 = Conv2D(32, (3, 3), padding='same', activation='relu')(inputs)
    branch1 = Conv2D(64, (3, 3), padding='same', activation='relu')(branch1)

    # Second Convolutional Branch (5x5 Convolutions)
    branch2 = Conv2D(32, (5, 5), padding='same', activation='relu')(inputs)
    branch2 = Conv2D(64, (5, 5), padding='same', activation='relu')(branch2)

    # Merge branches with element-wise addition
    merged = Add()([branch1, branch2])

    # Global Average Pooling Layer
    pooled = GlobalAveragePooling2D()(merged)

    # Fully Connected Layers
    dense1 = Dense(128, activation='relu')(pooled)
    dense2 = Dense(10)(dense1)  # 10 classes for CIFAR-10

    # Softmax layer for class probabilities
    outputs = Softmax()(dense2)

    # Create the model
    model = Model(inputs=inputs, outputs=outputs)

    # Compile the model
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    return model

# Example of how to use the model
if __name__ == "__main__":
    # Load CIFAR-10 dataset
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    
    # Preprocess the data
    x_train = x_train.astype('float32') / 255.0
    x_test = x_test.astype('float32') / 255.0
    y_train = to_categorical(y_train, num_classes=10)
    y_test = to_categorical(y_test, num_classes=10)
    
    # Create the model
    model = dl_model()
    
    # Train the model
    model.fit(x_train, y_train, batch_size=64, epochs=10, validation_data=(x_test, y_test))