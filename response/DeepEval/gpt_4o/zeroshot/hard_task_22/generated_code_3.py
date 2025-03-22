import tensorflow as tf
from tensorflow.keras.layers import Input, Lambda, SeparableConv2D, Conv2D, Add, Flatten, Dense, Concatenate
from tensorflow.keras.models import Model
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.utils import to_categorical

def dl_model(input_shape=(32, 32, 3), num_classes=10):
    # Input layer
    inputs = Input(shape=input_shape)
    
    # Split input into three groups along the channel dimension
    def split_input(x):
        return tf.split(x, num_or_size_splits=3, axis=-1)

    split_layers = Lambda(split_input)(inputs)

    # Main path with multi-scale feature extraction
    conv_1x1 = SeparableConv2D(filters=32, kernel_size=(1, 1), activation='relu')(split_layers[0])
    conv_3x3 = SeparableConv2D(filters=32, kernel_size=(3, 3), padding='same', activation='relu')(split_layers[1])
    conv_5x5 = SeparableConv2D(filters=32, kernel_size=(5, 5), padding='same', activation='relu')(split_layers[2])

    # Concatenate outputs of different kernel sizes
    main_path_output = Concatenate(axis=-1)([conv_1x1, conv_3x3, conv_5x5])

    # Branch path with 1x1 convolution to align channel dimensions
    branch_path_output = Conv2D(filters=96, kernel_size=(1, 1), activation='relu')(inputs)

    # Fuse main and branch paths
    fused_output = Add()([main_path_output, branch_path_output])

    # Flatten and add fully connected layers
    x = Flatten()(fused_output)
    x = Dense(128, activation='relu')(x)
    outputs = Dense(num_classes, activation='softmax')(x)

    # Create model
    model = Model(inputs=inputs, outputs=outputs)

    return model

# Load CIFAR-10 data
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)

# Create the model
model = dl_model()

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(x_train, y_train, batch_size=64, epochs=10, validation_data=(x_test, y_test))

# Evaluate the model
model.evaluate(x_test, y_test)