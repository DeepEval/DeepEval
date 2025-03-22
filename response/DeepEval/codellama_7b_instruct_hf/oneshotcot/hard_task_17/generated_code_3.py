import keras
from keras.layers import Input, GlobalAveragePooling2D, Flatten, Dense, Conv2D, MaxPooling2D, Add
from keras.models import Model

# Define input shape
input_shape = (32, 32, 3)

# Define the first block
def block1(input_tensor):
    # Global average pooling
    pool = GlobalAveragePooling2D()(input_tensor)
    # Reshape weights
    weights = Dense(units=input_shape[3], activation='relu')(pool)
    weights = Reshape(target_shape=input_shape[1:])(weights)
    # Multiply weights with input
    output = multiply([input_tensor, weights])
    return output

# Define the second block
def block2(input_tensor):
    # Convolutional layers
    conv1 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_tensor)
    conv2 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(conv1)
    # Max pooling layer
    pool = MaxPooling2D(pool_size=(2, 2), strides=2, padding='same')(conv2)
    return pool

# Define the input layer
input_layer = Input(shape=input_shape)

# Define the first block
block1_output = block1(input_layer)

# Define the second block
block2_output = block2(block1_output)

# Define the branch from Block 1
branch_output = block1(block2_output)

# Add the branch output to the main path output
add_output = Add()([block2_output, branch_output])

# Define the fully connected layers
flatten_layer = Flatten()(add_output)
dense1 = Dense(units=128, activation='relu')(flatten_layer)
dense2 = Dense(units=64, activation='relu')(dense1)
output_layer = Dense(units=10, activation='softmax')(dense2)

# Create the model
model = Model(inputs=input_layer, outputs=output_layer)