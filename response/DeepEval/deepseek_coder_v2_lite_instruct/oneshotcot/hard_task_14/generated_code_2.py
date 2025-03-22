import keras
from keras.layers import Input, Conv2D, GlobalAveragePooling2D, Reshape, Multiply, Add, Dense, Flatten
from keras.models import Model

def dl_model():
    input_layer = Input(shape=(32, 32, 3))  # CIFAR-10 images are 32x32 with 3 color channels

    # Main path
    main_path = Conv2D(filters=32, kernel_size=(3, 3), padding='same', activation='relu')(input_layer)
    main_path = Conv2D(filters=32, kernel_size=(3, 3), padding='same', activation='relu')(main_path)
    main_path = GlobalAveragePooling2D()(main_path)
    main_path = Reshape((1, 1, 32))(main_path)  # Reshape to match the number of channels in the input
    main_path = Multiply()([input_layer, main_path])  # Element-wise multiplication

    # Branch path
    branch_path = Conv2D(filters=32, kernel_size=(3, 3), padding='same', activation='relu')(input_layer)
    branch_path = Conv2D(filters=32, kernel_size=(3, 3), padding='same', activation='relu')(branch_path)
    branch_path = Conv2D(filters=32, kernel_size=(3, 3), padding='same', activation='relu')(branch_path)

    # Combine paths
    combined = Add()([main_path, branch_path])

    # Flatten and add dense layers
    combined = Flatten()(combined)
    combined = Dense(units=256, activation='relu')(combined)
    combined = Dense(units=128, activation='relu')(combined)
    output_layer = Dense(units=10, activation='softmax')(combined)

    model = Model(inputs=input_layer, outputs=output_layer)

    return model