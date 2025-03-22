import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, BatchNormalization, Flatten, Dense, Lambda
from keras.models import Model
from keras.layers import add

def dl_model():
    # Define the input shape (CIFAR-10 images are 32x32x3)
    input_layer = Input(shape=(32, 32, 3))

    # Block 1: Dual-path structure
    def block1(input_tensor):
        # Main path
        conv1 = Conv2D(filters=32, kernel_size=(3, 3), padding='same', activation='relu')(input_tensor)
        conv1 = Conv2D(filters=64, kernel_size=(1, 1), padding='same')(conv1)  # Restore the number of channels

        # Branch path
        branch = Conv2D(filters=64, kernel_size=(1, 1), padding='same')(input_tensor)

        # Combine the paths
        combined = add([conv1, branch])
        return combined

    block1_output = block1(input_layer)
    block1_output = BatchNormalization()(block1_output)
    block1_output = Flatten()(block1_output)

    # Block 2: Split input into three groups and process each with depthwise separable convolutional layers
    def block2(input_tensor):
        # Split the input into three groups
        split = Lambda(lambda x: tf.split(x, 3, axis=1))(input_tensor)

        # Process each group
        group1 = Conv2D(filters=64, kernel_size=(1, 1), activation='relu')(split[0])
        group2 = Conv2D(filters=64, kernel_size=(3, 3), activation='relu')(split[1])
        group3 = Conv2D(filters=64, kernel_size=(5, 5), activation='relu')(split[2])

        # Concatenate the outputs from the three groups
        output_tensor = Concatenate(axis=1)([group1, group2, group3])
        return output_tensor

    block2_output = block2(block1_output)
    block2_output = BatchNormalization()(block2_output)
    block2_output = Flatten()(block2_output)

    # Fully connected layers for classification
    dense1 = Dense(units=512, activation='relu')(block2_output)
    dense2 = Dense(units=256, activation='relu')(dense1)
    output_layer = Dense(units=10, activation='softmax')(dense2)

    # Construct the model
    model = Model(inputs=input_layer, outputs=output_layer)

    return model

# Instantiate the model
model = dl_model()

# Print the model summary
model.summary()