from tensorflow.keras.layers import Input, Lambda, DepthwiseConv2D, Conv2D, Concatenate, Add, Flatten, Dense
from tensorflow.keras.models import Model
import tensorflow as tf
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.utils import to_categorical

def dl_model():
    # Input shape for CIFAR-10 images
    input_shape = (32, 32, 3)
    
    # Input layer
    inputs = Input(shape=input_shape)

    # Split the input channels into 3 groups
    split_channels = Lambda(lambda x: tf.split(x, num_or_size_splits=3, axis=-1))(inputs)

    # Main path with depthwise separable convolutions
    conv_1x1 = DepthwiseConv2D(kernel_size=(1, 1), padding='same', activation='relu')(split_channels[0])
    conv_3x3 = DepthwiseConv2D(kernel_size=(3, 3), padding='same', activation='relu')(split_channels[1])
    conv_5x5 = DepthwiseConv2D(kernel_size=(5, 5), padding='same', activation='relu')(split_channels[2])
    
    # Concatenate the outputs
    main_path_output = Concatenate()([conv_1x1, conv_3x3, conv_5x5])

    # Branch path
    branch_path = Conv2D(filters=main_path_output.shape[-1], kernel_size=(1, 1), padding='same', activation='relu')(inputs)

    # Combine main and branch paths
    combined_output = Add()([main_path_output, branch_path])

    # Flatten the output for the dense layers
    flattened = Flatten()(combined_output)

    # Fully connected layers for classification
    dense1 = Dense(128, activation='relu')(flattened)
    dense2 = Dense(10, activation='softmax')(dense1)

    # Create the model
    model = Model(inputs=inputs, outputs=dense2)

    return model

# Instantiate the model
model = dl_model()

# Print model summary
model.summary()

# Load CIFAR-10 dataset
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

# Normalize the input images
x_train = x_train.astype('float32') / 255
x_test = x_test.astype('float32') / 255

# One-hot encode the labels
y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(x_train, y_train, batch_size=64, epochs=10, validation_split=0.1)