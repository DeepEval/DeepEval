import keras
from keras.layers import Input, MaxPooling2D, Flatten, Dense, Reshape, Conv2D, Concatenate
from keras.models import Model

def dl_model():
    # Input layer
    input_layer = Input(shape=(28, 28, 1))

    # Block 1: Three max pooling layers with different scales
    pool1 = MaxPooling2D(pool_size=(1, 1), strides=(1, 1), padding='same')(input_layer)
    pool2 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(input_layer)
    pool3 = MaxPooling2D(pool_size=(4, 4), strides=(4, 4), padding='same')(input_layer)

    # Flatten each pooled result
    flat1 = Flatten()(pool1)
    flat2 = Flatten()(pool2)
    flat3 = Flatten()(pool3)

    # Concatenate flattened outputs
    concat1 = Concatenate()([flat1, flat2, flat3])

    # Between Block 1 and Block 2: Fully connected layer and reshape
    fc1 = Dense(units=128, activation='relu')(concat1)
    reshape_layer = Reshape(target_shape=(4, 4, 8))(fc1)  # Adjust target shape to match downstream layers

    # Block 2: Multiple branches with convolutions and pooling
    conv1x1 = Conv2D(filters=16, kernel_size=(1, 1), activation='relu', padding='same')(reshape_layer)
    conv3x3 = Conv2D(filters=32, kernel_size=(3, 3), activation='relu', padding='same')(reshape_layer)
    conv5x5 = Conv2D(filters=48, kernel_size=(5, 5), activation='relu', padding='same')(reshape_layer)
    pool3x3 = MaxPooling2D(pool_size=(3, 3), strides=(1, 1), padding='same')(reshape_layer)

    # Concatenate outputs from all branches
    concat2 = Concatenate()([conv1x1, conv3x3, conv5x5, pool3x3])

    # Final classification: Flatten and fully connected layer
    flat_final = Flatten()(concat2)
    output_layer = Dense(units=10, activation='softmax')(flat_final)

    # Construct the model
    model = Model(inputs=input_layer, outputs=output_layer)

    return model

# Create the model
model = dl_model()
model.summary()  # This will print the model summary to verify the architecture