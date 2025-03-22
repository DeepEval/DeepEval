import keras
from keras.layers import Input, Conv2D, BatchNormalization, ReLU, Add, GlobalAveragePooling2D, Flatten, Dense
from keras.models import Model

def dl_model():
    input_layer = Input(shape=(32, 32, 3))  # CIFAR-10 images are 32x32 with 3 color channels

    # Initial convolutional layer to reduce dimensionality
    x = Conv2D(filters=16, kernel_size=(3, 3), strides=(1, 1), padding='same')(input_layer)
    x = BatchNormalization()(x)
    x = ReLU()(x)

    # Define a basic block
    def basic_block(x):
        # Main path
        main_path = Conv2D(filters=16, kernel_size=(3, 3), strides=(1, 1), padding='same')(x)
        main_path = BatchNormalization()(main_path)
        main_path = ReLU()(main_path)

        # Branch path
        branch_path = Conv2D(filters=16, kernel_size=(3, 3), strides=(1, 1), padding='same')(x)
        branch_path = BatchNormalization()(branch_path)
        branch_path = ReLU()(branch_path)

        # Add the outputs of both paths
        output_tensor = Add()([main_path, branch_path])
        output_tensor = ReLU()(output_tensor)

        return output_tensor

    # Apply the basic block twice
    x = basic_block(x)
    x = basic_block(x)

    # Global average pooling
    x = GlobalAveragePooling2D()(x)

    # Flatten the feature map
    x = Flatten()(x)

    # Fully connected layer
    output_layer = Dense(units=10, activation='softmax')(x)

    model = Model(inputs=input_layer, outputs=output_layer)

    return model