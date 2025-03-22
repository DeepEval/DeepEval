from tensorflow.keras import layers, models, Input
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.utils import to_categorical

def dl_model():
    input_shape = (32, 32, 3)  # CIFAR-10 image dimensions
    num_classes = 10  # CIFAR-10 has 10 classes

    # Define the input layer
    inputs = Input(shape=input_shape)

    # Block 1 - Parallel branches
    conv1x1 = layers.Conv2D(32, (1, 1), padding='same', activation='relu')(inputs)
    conv3x3 = layers.Conv2D(32, (3, 3), padding='same', activation='relu')(inputs)
    conv5x5 = layers.Conv2D(32, (5, 5), padding='same', activation='relu')(inputs)
    max_pool = layers.MaxPooling2D((3, 3), strides=(1, 1), padding='same')(inputs)
    
    # Concatenate all branches
    concat = layers.concatenate([conv1x1, conv3x3, conv5x5, max_pool], axis=-1)

    # Block 2 - Global Average Pooling and Fully Connected Layers
    gap = layers.GlobalAveragePooling2D()(concat)
    
    # Fully connected layers to produce weights
    dense1 = layers.Dense(128, activation='relu')(gap)
    dense2 = layers.Dense(concat.shape[-1], activation='sigmoid')(dense1)  # Output weights

    # Reshape weights to match input feature map shape
    weights = layers.Reshape((1, 1, concat.shape[-1]))(dense2)
    
    # Element-wise multiplication with the input feature map of block 2
    scaled_features = layers.Multiply()([concat, weights])

    # Final fully connected layer to produce output probability distribution
    output = layers.Flatten()(scaled_features)
    output = layers.Dense(num_classes, activation='softmax')(output)

    # Define the model
    model = models.Model(inputs=inputs, outputs=output)

    return model

# Load CIFAR-10 data
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

# Preprocess data
x_train = x_train.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0
y_train = to_categorical(y_train, num_classes=10)
y_test = to_categorical(y_test, num_classes=10)

# Get the model
model = dl_model()

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Summary of the model
model.summary()

# (Optional) Train the model
model.fit(x_train, y_train, batch_size=64, epochs=10, validation_split=0.2)

# (Optional) Evaluate the model
model.evaluate(x_test, y_test)