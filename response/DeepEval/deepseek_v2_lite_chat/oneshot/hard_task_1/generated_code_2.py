import keras
from keras.datasets import cifar10
from keras.models import Model
from keras.layers import Input, Conv2D, GlobalAveragePooling2D, GlobalMaxPooling2D, Concatenate, Dense, Activation
from keras.layers import BatchNormalization, Flatten, MaxPooling2D, Concatenate, Activation
from keras.layers.advanced_activations import LeakyReLU

def dl_model():
    # Number of classes
    NUM_CLASSES = 10

    # Input shape
    input_shape = (32, 32, 3)  # CIFAR-10 images are 32x32 pixels

    # Load CIFAR-10 data
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()

    # Normalize pixel values to be between 0 and 1
    x_train, x_test = x_train / 255.0, x_test / 255.0

    # Model inputs
    input_layer = Input(shape=input_shape)

    # Adjust the number of output channels to match the input image channels
    conv = Conv2D(32, (3, 3), padding='same')(input_layer)

    # Block 1: Parallel paths for feature extraction
    def block1():
        # Path1: Global average pooling followed by dense layers
        avg_pool = GlobalAveragePooling2D()(conv)
        dense1 = Dense(256, activation='relu')(avg_pool)
        dense2 = Dense(128, activation='relu')(dense1)

        # Path2: Global max pooling followed by dense layers
        max_pool = GlobalMaxPooling2D()(conv)
        dense3 = Dense(256, activation='relu')(max_pool)
        dense4 = Dense(128, activation='relu')(dense3)

        # Combine outputs from both paths
        combined = Concatenate()([dense1, dense2, dense3, dense4])
        attention = Dense(1)(combined)
        attention = Activation('sigmoid')(attention)  # Sigmoid activation to generate attention weights
        combined = combined * attention  # Apply attention weights

        return combined

    block1_output = block1()

    # Block 2: Spatial feature extraction
    def block2():
        # Average pooling
        avg_pool = GlobalAveragePooling2D()(conv)
        dense5 = Dense(256, activation='relu')(avg_pool)
        dense6 = Dense(128, activation='relu')(dense5)

        # Max pooling
        max_pool = GlobalMaxPooling2D()(conv)
        dense7 = Dense(256, activation='relu')(max_pool)
        dense8 = Dense(128, activation='relu')(dense7)

        # Concatenate along the channel dimension
        concat = Concatenate()([dense5, dense6, dense7, dense8])

        # 1x1 convolution and sigmoid activation for normalization
        conv1x1 = Conv2D(1, (1, 1), padding='same')(concat)
        conv1x1 = Activation('sigmoid')(conv1x1)

        # Element-wise multiplication with the combined output from Block 1
        output = Concatenate()([combined, conv1x1])

        return output

    block2_output = block2()

    # Output layer
    output_layer = Dense(NUM_CLASSES, activation='softmax')(block2_output)

    # Model construction
    model = Model(inputs=input_layer, outputs=output_layer)

    return model

# Create and compile the model
model = dl_model()
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Print model summary
model.summary()