from tensorflow.keras.layers import Input, Conv2D, GlobalAveragePooling2D, Dense, Multiply, Reshape, Add, Flatten
from tensorflow.keras.models import Model
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.utils import to_categorical

def dl_model():
    # Define input
    input_shape = (32, 32, 3)
    inputs = Input(shape=input_shape)
    
    # Main path: Global Average Pooling -> Fully Connected -> Reshape -> Multiply
    x = GlobalAveragePooling2D()(inputs)
    x = Dense(units=64, activation='relu')(x)  # Adjust number of units as per the main feature map channels
    x = Dense(units=input_shape[-1], activation='sigmoid')(x)  # To generate weights with same channels as input
    x = Reshape((1, 1, input_shape[-1]))(x)  # Reshape to (1, 1, channels)
    x = Multiply()([inputs, x])  # Element-wise multiplication with the original input

    # Branch path: Convolution
    y = Conv2D(filters=input_shape[-1], kernel_size=(3, 3), padding='same', activation='relu')(inputs)
    
    # Combine paths
    combined = Add()([x, y])
    
    # Fully connected layers for classification
    combined = Flatten()(combined)
    combined = Dense(units=128, activation='relu')(combined)
    combined = Dense(units=64, activation='relu')(combined)
    outputs = Dense(units=10, activation='softmax')(combined)
    
    # Create model
    model = Model(inputs=inputs, outputs=outputs)
    
    # Compile the model
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    
    return model

# Prepare the CIFAR-10 data
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)

# Create the model
model = dl_model()

# Print the model summary
model.summary()

# Train the model
# Note: You might need to adjust batch_size and epochs according to your environment
model.fit(x_train, y_train, batch_size=64, epochs=10, validation_data=(x_test, y_test))