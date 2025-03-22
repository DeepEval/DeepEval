import keras
from keras.datasets import cifar10
from keras.layers import Input, Conv2D, MaxPooling2D, Add, BatchNormalization, Flatten, Dense, GlobalAveragePooling2D

def dl_model():
    # Load CIFAR-10 dataset
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()

    # Normalize the pixel values to be between 0 and 1
    x_train = x_train / 255.0
    x_test = x_test / 255.0

    # Define the input shape
    input_shape = (32, 32, 3)  # CIFAR-10 images are 32x32

    # Input layer
    input_layer = Input(shape=input_shape)

    # Adjust the feature dimensionality to 16 using a convolutional layer
    conv1 = Conv2D(filters=16, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_layer)

    # Basic block with convolution, batch normalization, and ReLU activation
    def basic_block(input_tensor):
        bn = BatchNormalization()(input_tensor)
        bn_relu = keras.activations.relu(bn)
        conv = Conv2D(filters=16, kernel_size=(3, 3), strides=(1, 1), padding='same')(bn_relu)
        return conv

    # First path of the basic block
    main_path = basic_block(input_tensor=conv1)

    # Branch connecting directly to the block's input
    branch = basic_block(input_tensor=input_layer)

    # Add the outputs of the main path and the branch
    add = Add()([main_path, branch])

    # Further processing of the output
    block_output = Flatten()(add)
    dense1 = Dense(units=128, activation='relu')(block_output)
    dense2 = Dense(units=64, activation='relu')(dense1)
    output_layer = Dense(units=10, activation='softmax')(dense2)

    # Residual connections
    def residual_block(input_tensor, filters, block_num):
        for _ in range(block_num):
            conv = Conv2D(filters=filters, kernel_size=(3, 3), strides=(1, 1), padding='same')(input_tensor)
            bn = BatchNormalization()(conv)
            bn_relu = keras.activations.relu(bn)
            input_tensor = bn_relu
        return input_tensor

    # Second-level residual block
    block2_output = residual_block(input_tensor=conv1, filters=16, block_num=2)

    # Global branch for capturing features from the initial convolution output
    global_branch = Conv2D(filters=16, kernel_size=(1, 1), strides=(1, 1), padding='same')(input_layer)
    global_branch = BatchNormalization()(global_branch)
    global_branch = keras.activations.relu(global_branch)
    global_branch = Conv2D(filters=16, kernel_size=(3, 3), strides=(1, 1), padding='same')(global_branch)

    # Add the outputs of the global branch to the second-level residual block
    level3_output = Add()([block2_output, global_branch])

    # Flatten and fully connected layers
    flat = Flatten()(level3_output)
    output = Dense(units=10, activation='softmax')(flat)

    # Model architecture
    model = keras.Model(inputs=input_layer, outputs=output)

    return model

# Instantiate the model
model = dl_model()

# Print model summary
model.summary()