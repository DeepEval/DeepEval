import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, BatchNormalization, Flatten, Dense, SeparableConv2D
from keras.models import Model

def dl_model():
    # Define the main path
    def main_path(input_tensor):
        # Block 1
        x = SeparableConv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        # Block 2
        x = SeparableConv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(x)
        # Block 3
        x = SeparableConv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(x)
        return x

    # Define the branch path
    def branch_path(input_tensor):
        x = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        return x

    # Input layer
    input_layer = Input(shape=(28, 28, 1))

    # Main path processing
    main_output = main_path(input_layer)

    # Branch path processing
    branch_output = branch_path(input_layer)

    # Concatenate the main path and branch path outputs along the channel dimension
    fused_features = Concatenate(axis=-1)([main_output, branch_output])

    # Flatten the fused features
    flattened_features = Flatten()(fused_features)

    # Fully connected layer for classification
    output_layer = Dense(units=10, activation='softmax')(flattened_features)

    # Construct the model
    model = Model(inputs=input_layer, outputs=output_layer)

    return model

# Build the model
model = dl_model()
model.summary()