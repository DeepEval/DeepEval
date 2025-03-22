import keras
from keras.layers import Input, Lambda, Conv2D, MaxPool2D, Add, Flatten, Dense, Reshape, Concatenate
from keras.models import Model

def dl_model():
    # Input layer
    input_layer = Input(shape=(32, 32, 3))

    # Block 1
    def block_1(input_tensor):
        # Split into three groups
        split_1 = Lambda(lambda x: tf.split(x, num_or_size_splits=3, axis=-1))(input_tensor)
        # Convolutional layers
        conv1_1 = Conv2D(filters=32, kernel_size=(1, 1), padding='same')(split_1[0])
        conv1_2 = Conv2D(filters=32, kernel_size=(3, 3), padding='same')(split_1[1])
        conv1_3 = Conv2D(filters=32, kernel_size=(1, 1), padding='same')(split_1[2])
        # Concatenate the outputs
        concat1 = Concatenate(axis=-1)([conv1_1, conv1_2, conv1_3])
        return concat1

    # Transition Convolution
    transition_conv = Conv2D(filters=64, kernel_size=(3, 3), strides=(2, 2), padding='same')(block_1(input_tensor=input_layer))

    # Block 2
    def block_2(input_tensor):
        # Global max pooling
        pool1 = MaxPool2D(pool_size=(3, 3), strides=(1, 1), padding='same')(input_tensor)
        # Fully connected layers for channel weights
        fc1 = Dense(units=512, activation='relu')(pool1)
        fc2 = Dense(units=128, activation='relu')(fc1)
        # Reshape weights to match output shape
        reshaped_weights = Reshape((1, 1, -1, 512))(fc2)
        # Multiply weights with pool1
        main_path_output = Conv2DTranspose(filters=64, kernel_size=(3, 3), strides=(2, 2), padding='same')(reshaped_weights)
        main_path_output = Conv2D(filters=64, kernel_size=(3, 3), padding='same', activation='relu')(main_path_output)
        # Branch
        branch_input = Conv2D(filters=64, kernel_size=(1, 1), padding='same')(input_tensor)
        branch_output = Conv2D(filters=64, kernel_size=(1, 1), padding='same')(branch_input)
        # Add main path and branch path
        add_layer = Add()([main_path_output, branch_output])
        # Fully connected layer for classification
        output = Flatten()(add_layer)
        output = Dense(units=10, activation='softmax')(output)
        # Model
        model = Model(inputs=input_layer, outputs=output)
        return model

    # Create the model
    model = block_2(input_tensor=input_layer)
    return model

# Instantiate the model
model = dl_model()