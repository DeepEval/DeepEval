from tensorflow.keras.layers import Input, Conv2D, BatchNormalization, ReLU, GlobalAveragePooling2D, Dense, Multiply, Concatenate, AveragePooling2D, Flatten
from tensorflow.keras.models import Model
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.utils import to_categorical

def dl_model(input_shape=(32, 32, 3), num_classes=10):
    # Input layer
    inputs = Input(shape=input_shape)

    # Initial convolutional layer
    x = Conv2D(32, (3, 3), padding='same', activation=None)(inputs)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    
    # Store the initial features
    initial_features = x

    # Global average pooling
    gap = GlobalAveragePooling2D()(x)
    
    # Fully connected layers to adjust dimensions
    fc1 = Dense(32, activation='relu')(gap)
    fc2 = Dense(32, activation='sigmoid')(fc1)
    
    # Reshape to match the initial feature dimensions
    scale = Multiply()([initial_features, fc2])
    
    # Concatenate weighted feature maps with the input layer
    concatenated = Concatenate()([inputs, scale])
    
    # Reduce dimensionality and downsample features
    x = Conv2D(32, (1, 1), padding='same', activation='relu')(concatenated)
    x = AveragePooling2D(pool_size=(2, 2))(x)
    
    # Flatten and fully connected layer for classification
    x = Flatten()(x)
    outputs = Dense(num_classes, activation='softmax')(x)
    
    # Create the model
    model = Model(inputs=inputs, outputs=outputs)
    
    return model

# Example usage
# Load CIFAR-10 data to test model structure
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)

model = dl_model()
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()