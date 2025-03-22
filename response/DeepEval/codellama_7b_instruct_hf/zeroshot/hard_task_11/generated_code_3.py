import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense

def dl_model():
    # Input layer
    input_layer = Input(shape=(32, 32, 3))

    # First parallel branch
    branch_1 = Conv2D(32, (1, 1), strides=(1, 1), padding='same')(input_layer)
    branch_1 = Conv2D(32, (1, 3), strides=(1, 1), padding='same')(branch_1)
    branch_1 = Conv2D(32, (3, 1), strides=(1, 1), padding='same')(branch_1)
    branch_1 = MaxPooling2D((2, 2), strides=(2, 2))(branch_1)

    # Second parallel branch
    branch_2 = Conv2D(32, (1, 1), strides=(1, 1), padding='same')(input_layer)
    branch_2 = Conv2D(32, (3, 3), strides=(1, 1), padding='same')(branch_2)
    branch_2 = Conv2D(32, (3, 3), strides=(1, 1), padding='same')(branch_2)
    branch_2 = MaxPooling2D((2, 2), strides=(2, 2))(branch_2)

    # Concatenate outputs from parallel branches
    merged_output = keras.layers.Concatenate(axis=1)([branch_1, branch_2])

    # Additive fusion with main pathway
    merged_output = keras.layers.Add()([merged_output, input_layer])

    # Final convolution layer
    merged_output = Conv2D(64, (1, 1), strides=(1, 1), padding='same')(merged_output)
    merged_output = Flatten()(merged_output)

    # Fully connected layers
    merged_output = Dense(128, activation='relu')(merged_output)
    merged_output = Dense(10, activation='softmax')(merged_output)

    # Create model
    model = keras.models.Model(inputs=input_layer, outputs=merged_output)

    return model