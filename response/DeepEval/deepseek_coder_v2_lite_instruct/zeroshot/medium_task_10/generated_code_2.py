import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, BatchNormalization, ReLU, Add, AveragePooling2D, Flatten, Dense

def basic_block(x, filters):
    """
    Defines a basic block with main path and shortcut path.
    """
    main_path = Conv2D(filters, (3, 3), padding='same')(x)
    main_path = BatchNormalization()(main_path)
    main_path = ReLU()(main_path)
    main_path = Conv2D(filters, (3, 3), padding='same')(main_path)
    main_path = BatchNormalization()(main_path)

    shortcut = Conv2D(filters, (1, 1), padding='same')(x)
    shortcut = BatchNormalization()(shortcut)

    output = Add()([main_path, shortcut])
    output = ReLU()(output)
    return output

def dl_model():
    """
    Constructs the CIFAR-10 classification model using the Functional API.
    """
    inputs = Input(shape=(32, 32, 3))

    # Initial convolution layer
    x = Conv2D(16, (3, 3), padding='same')(inputs)
    x = BatchNormalization()(x)
    x = ReLU()(x)

    # First level of the core architecture
    x = basic_block(x, 16)

    # Second level of the core architecture
    x = basic_block(x, 16)
    x = basic_block(x, 16)

    # Third level of the core architecture
    x = basic_block(x, 32)
    x = basic_block(x, 32)
    global_branch = Conv2D(64, (3, 3), padding='same')(x)
    global_branch = BatchNormalization()(global_branch)
    global_branch = ReLU()(global_branch)

    # Adding global branch to the second level
    x = Add()([x, global_branch])

    # Final layers
    x = AveragePooling2D((8, 8))(x)
    x = Flatten()(x)
    outputs = Dense(10, activation='softmax')(x)

    model = Model(inputs, outputs)
    return model

# Example usage:
# model = dl_model()
# model.summary()