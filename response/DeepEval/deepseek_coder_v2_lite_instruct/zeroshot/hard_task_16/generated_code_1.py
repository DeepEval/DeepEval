import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, BatchNormalization, Activation, Add, Lambda, Dense, Reshape

def dl_model():
    # Define the input layer
    inputs = Input(shape=(32, 32, 3))

    # Block 1: Split the input into three groups and process each group
    def block1(x):
        # Split the input into three groups
        split_1, split_2, split_3 = tf.split(x, 3, axis=-1)

        # Process each group
        conv1_1 = Conv2D(32, (1, 1), activation='relu')(split_1)
        conv3_1 = Conv2D(32, (3, 3), padding='same', activation='relu')(conv1_1)
        conv1_2 = Conv2D(32, (1, 1), activation='relu')(conv3_1)

        conv1_3 = Conv2D(32, (1, 1), activation='relu')(split_2)
        conv3_2 = Conv2D(32, (3, 3), padding='same', activation='relu')(conv1_3)
        conv1_4 = Conv2D(32, (1, 1), activation='relu')(conv3_2)

        conv1_5 = Conv2D(32, (1, 1), activation='relu')(split_3)
        conv3_3 = Conv2D(32, (3, 3), padding='same', activation='relu')(conv1_5)
        conv1_6 = Conv2D(32, (1, 1), activation='relu')(conv3_3)

        # Concatenate the outputs
        output = tf.concat([conv1_2, conv1_4, conv1_6], axis=-1)
        return output

    # Apply Block 1
    x = block1(inputs)

    # Transition Convolution: Adjust the number of channels to match the input layer
    x = Conv2D(64, (1, 1), activation='relu')(x)

    # Block 2: Perform global max pooling and adjust weights
    def block2(x):
        # Global max pooling
        pooled = MaxPooling2D((8, 8))(x)

        # Generate channel-matching weights
        fc1 = Dense(32, activation='relu')(pooled)
        fc2 = Dense(x.shape[-1])(fc1)

        # Reshape the weights to match the shape of the adjusted output
        weights = Reshape((x.shape[1], x.shape[2], x.shape[3]))(fc2)

        # Multiply the weights with the adjusted output
        output = tf.multiply(weights, x)
        return output

    # Apply Block 2
    main_path = block2(x)

    # Branch of the model connecting directly to the input
    branch = Conv2D(64, (1, 1), activation='relu')(inputs)

    # Add the outputs from both the main path and the branch
    added = Add()([main_path, branch])

    # Final output: Fully connected layer for classification
    final_output = Dense(10, activation='softmax')(added)

    # Define the model
    model = Model(inputs=inputs, outputs=final_output)

    return model

# Example usage
model = dl_model()
model.summary()