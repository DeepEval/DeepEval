from tensorflow.keras.layers import Input, Conv2D, AveragePooling2D, Conv2DTranspose, Concatenate, Add, Flatten, Dense
from tensorflow.keras.models import Model

def dl_model():
    # Define the input shape for CIFAR-10 images
    input_shape = (32, 32, 3)
    inputs = Input(shape=input_shape)

    # Main path - initial 1x1 convolution
    x = Conv2D(64, (1, 1), activation='relu')(inputs)

    # First branch: 3x3 convolution
    branch1 = Conv2D(64, (3, 3), padding='same', activation='relu')(x)

    # Second branch: average pooling -> 3x3 convolution -> transpose convolution
    branch2 = AveragePooling2D((2, 2))(x)
    branch2 = Conv2D(64, (3, 3), padding='same', activation='relu')(branch2)
    branch2 = Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same', activation='relu')(branch2)

    # Third branch: average pooling -> 3x3 convolution -> transpose convolution
    branch3 = AveragePooling2D((2, 2))(x)
    branch3 = Conv2D(64, (3, 3), padding='same', activation='relu')(branch3)
    branch3 = Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same', activation='relu')(branch3)

    # Concatenate branches
    concatenated = Concatenate()([branch1, branch2, branch3])

    # Main path output: 1x1 convolution
    main_path_output = Conv2D(128, (1, 1), activation='relu')(concatenated)

    # Branch path: 1x1 convolution to match number of channels
    branch_path_output = Conv2D(128, (1, 1), activation='relu')(inputs)

    # Fuse main path and branch path outputs through addition
    fused_output = Add()([main_path_output, branch_path_output])

    # Flatten and fully connected layer for 10-class classification
    x = Flatten()(fused_output)
    outputs = Dense(10, activation='softmax')(x)

    # Create model
    model = Model(inputs=inputs, outputs=outputs)

    return model

# To get the model
model = dl_model()
model.summary()