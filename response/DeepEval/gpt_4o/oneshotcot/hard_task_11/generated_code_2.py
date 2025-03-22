import keras
from keras.layers import Input, Conv2D, Concatenate, Add, Flatten, Dense
from keras.models import Model

def dl_model():
    # Input layer
    input_layer = Input(shape=(32, 32, 3))

    # Main pathway with two parallel branches
    # First path: 1x1 Convolution
    path1 = Conv2D(filters=32, kernel_size=(1, 1), activation='relu', padding='same')(input_layer)

    # Second path: 1x1, 1x3, and 3x1 Convolutions
    path2_1x1 = Conv2D(filters=32, kernel_size=(1, 1), activation='relu', padding='same')(input_layer)
    path2_1x3 = Conv2D(filters=32, kernel_size=(1, 3), activation='relu', padding='same')(path2_1x1)
    path2_3x1 = Conv2D(filters=32, kernel_size=(3, 1), activation='relu', padding='same')(path2_1x3)

    # Concatenate outputs from the two paths
    concatenated = Concatenate()([path1, path2_3x1])

    # Another 1x1 convolution to match the input channel dimensions
    main_output = Conv2D(filters=3, kernel_size=(1, 1), activation='relu', padding='same')(concatenated)

    # Add input directly to the main output (Residual Connection)
    fused_output = Add()([input_layer, main_output])

    # Flatten the final feature map
    flatten_layer = Flatten()(fused_output)

    # Fully connected layers for classification
    dense1 = Dense(units=128, activation='relu')(flatten_layer)
    output_layer = Dense(units=10, activation='softmax')(dense1)

    # Create the model
    model = Model(inputs=input_layer, outputs=output_layer)

    return model