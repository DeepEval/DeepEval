import keras
from keras.layers import Input, Conv2D, GlobalAveragePooling2D, Dense, Add, Reshape
from keras.models import Model

def dl_model():
    input_layer = Input(shape=(32, 32, 3))  # CIFAR-10 images are 32x32 with 3 color channels

    # Main path
    main_path = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_layer)
    main_path = GlobalAveragePooling2D()(main_path)
    main_path = Dense(units=32, activation='relu')(main_path)  # Number of units should match the number of channels in the input
    main_path = Reshape((1, 1, 32))(main_path)  # Reshape to match the input shape
    main_path = Multiply()([input_layer, main_path])  # Element-wise multiplication

    # Branch path
    branch_path = input_layer

    # Combine both paths
    combined = Add()([main_path, branch_path])

    # Further processing
    combined = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(combined)
    combined = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(combined)
    combined = GlobalAveragePooling2D()(combined)
    output_layer = Dense(units=10, activation='softmax')(combined)

    model = Model(inputs=input_layer, outputs=output_layer)
    return model