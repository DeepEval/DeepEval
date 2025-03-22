import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, BatchNormalization, Flatten, Dense, Lambda, Dense, Add
from tensorflow.keras.models import Model

def dl_model():
    input_layer = Input(shape=(32, 32, 3))

    # Block 1
    def block1(x):
        # Split the input into three groups
        split1 = Lambda(lambda tensor: tensor[:, :, :, :tensor.shape[3]//3])(x)
        split2 = Lambda(lambda tensor: tensor[:, :, :, tensor.shape[3]//3:2*tensor.shape[3]//3])(x)
        split3 = Lambda(lambda tensor: tensor[:, :, :, 2*tensor.shape[3]//3:])(x)

        # Process each group
        processed1 = Conv2D(64, (1, 1), activation='relu')(split1)
        processed1 = Conv2D(64, (3, 3), activation='relu')(processed1)
        processed1 = Conv2D(64, (1, 1), activation='relu')(processed1)

        processed2 = Conv2D(64, (1, 1), activation='relu')(split2)
        processed2 = Conv2D(64, (3, 3), activation='relu')(processed2)
        processed2 = Conv2D(64, (1, 1), activation='relu')(processed2)

        processed3 = Conv2D(64, (1, 1), activation='relu')(split3)
        processed3 = Conv2D(64, (3, 3), activation='relu')(processed3)
        processed3 = Conv2D(64, (1, 1), activation='relu')(processed3)

        # Concatenate the outputs
        concatenated = Concatenate()([processed1, processed2, processed3])
        return concatenated

    block1_output = block1(input_layer)

    # Transition Convolution
    transition = Conv2D(32, (1, 1), activation='relu')(block1_output)

    # Block 2
    def block2(x):
        # Global max pooling
        pooled = MaxPooling2D((8, 8))(x)

        # Fully connected layers to generate weights
        fc1 = Dense(128, activation='relu')(pooled)
        fc2 = Dense(x.shape[3], activation='relu')(fc1)

        # Reshape weights to match the shape of the input
        weights = tf.reshape(fc2, (1, 1, 1, x.shape[3]))

        # Multiply weights with the input
        weighted_output = tf.multiply(x, weights)
        return weighted_output

    block2_output = block2(transition)

    # Direct branch from input
    branch = input_layer

    # Add the main path and the branch
    added = Add()([block2_output, branch])

    # Flatten the result
    flattened = Flatten()(added)

    # Fully connected layers for classification
    dense1 = Dense(128, activation='relu')(flattened)
    dense2 = Dense(64, activation='relu')(dense1)
    output_layer = Dense(10, activation='softmax')(dense2)

    model = Model(inputs=input_layer, outputs=output_layer)
    return model

# Build the model
model = dl_model()
model.summary()