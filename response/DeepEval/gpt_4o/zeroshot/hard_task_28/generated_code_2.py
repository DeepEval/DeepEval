from tensorflow.keras.layers import Input, DepthwiseConv2D, LayerNormalization, Conv2D, Add, Flatten, Dense
from tensorflow.keras.models import Model

def dl_model():
    # Input layer for CIFAR-10 images
    input_layer = Input(shape=(32, 32, 3))

    # Main path
    # 1. Depthwise Convolution 7x7
    main_path = DepthwiseConv2D(kernel_size=(7, 7), padding='same')(input_layer)
    # 2. Layer Normalization
    main_path = LayerNormalization()(main_path)
    # 3. First 1x1 Pointwise Convolution
    main_path = Conv2D(filters=3, kernel_size=(1, 1), padding='same')(main_path)
    # 4. Second 1x1 Pointwise Convolution
    main_path = Conv2D(filters=3, kernel_size=(1, 1), padding='same')(main_path)

    # Branch path (direct connection to input)
    branch_path = input_layer

    # Combine paths by addition
    combined_output = Add()([main_path, branch_path])

    # Flatten the combined output
    flatten_layer = Flatten()(combined_output)

    # Fully connected layers for classification
    fc1 = Dense(units=256, activation='relu')(flatten_layer)
    fc2 = Dense(units=10, activation='softmax')(fc1)  # CIFAR-10 has 10 classes

    # Construct the model
    model = Model(inputs=input_layer, outputs=fc2)

    return model