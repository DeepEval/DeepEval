import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Dropout, Concatenate, Lambda, Flatten, Dense

def dl_model():
    # Input layer
    input_layer = Input(shape=(32, 32, 3))

    # Block 1: Feature extraction with varying kernel sizes
    split = Lambda(lambda x: tf.split(x, 3, axis=-1))(input_layer)
    
    # Extract features with 1x1, 3x3, and 5x5 kernels
    conv1x1 = Conv2D(32, (1, 1), activation='relu')(split[0])
    conv3x3 = Conv2D(32, (3, 3), activation='relu', padding='same')(split[1])
    conv5x5 = Conv2D(32, (5, 5), activation='relu', padding='same')(split[2])
    
    # Dropout layer to reduce overfitting
    dropout = Dropout(0.25)(conv5x5)
    
    # Concatenate the outputs
    concat = Concatenate(axis=-1)([conv1x1, conv3x3, dropout])

    # Block 2: Multiple branches for feature processing and fusion
    branch1 = Conv2D(32, (1, 1), activation='relu')(concat)
    branch2 = Conv2D(32, (1, 1), activation='relu')(concat)
    branch3 = Conv2D(32, (3, 3), activation='relu', padding='same')(concat)
    branch4 = Conv2D(32, (5, 5), activation='relu', padding='same')(concat)
    branch5 = Conv2D(32, (1, 1), activation='relu')(MaxPooling2D((3, 3))(concat))

    # Concatenate outputs from all branches
    concat_branches = Concatenate(axis=-1)([branch1, branch2, branch3, branch4, branch5])

    # Flatten the output
    flatten = Flatten()(concat_branches)

    # Fully connected layer
    dense = Dense(128, activation='relu')(flatten)
    output_layer = Dense(10, activation='softmax')(dense)

    # Define the model
    model = Model(inputs=input_layer, outputs=output_layer)

    return model

# Example usage:
# model = dl_model()
# model.summary()