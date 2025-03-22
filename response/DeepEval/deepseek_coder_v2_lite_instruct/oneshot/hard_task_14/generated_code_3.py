import keras
from keras.layers import Input, Conv2D, GlobalAveragePooling2D, Concatenate, BatchNormalization, Flatten, Dense, Multiply
from keras.models import Model

def dl_model():
    input_layer = Input(shape=(32, 32, 3))  # CIFAR-10 images are 32x32 with 3 channels (RGB)

    # Main path
    main_path = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_layer)
    main_path = BatchNormalization()(main_path)
    main_path = GlobalAveragePooling2D()(main_path)
    main_path = Dense(units=64, activation='relu')(main_path)  # Adjusting to match the number of channels
    main_path = Dense(units=32, activation='relu')(main_path)  # Adjusting to match the number of channels
    main_weights = Dense(units=32, activation='sigmoid')(main_path)  # Generating weights
    main_weights = keras.backend.reshape(main_weights, (1, 1, 32))  # Reshaping to match input shape
    main_path = Multiply()([main_weights, input_layer])  # Element-wise multiplication

    # Branch path
    branch_path = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_layer)

    # Combine paths
    combined = Concatenate()([main_path, branch_path])

    # Flatten and fully connected layers
    combined = Flatten()(combined)
    fc1 = Dense(units=128, activation='relu')(combined)
    fc2 = Dense(units=64, activation='relu')(fc1)
    output_layer = Dense(units=10, activation='softmax')(fc2)

    model = Model(inputs=input_layer, outputs=output_layer)

    return model