import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, BatchNormalization, Flatten, Dense

def dl_model():
    # Input layer
    input_layer = Input(shape=(28, 28, 1))

    # Main path
    main_path = input_layer
    for i in range(3):
        main_path = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(main_path)
        main_path = MaxPooling2D(pool_size=(2, 2), strides=2, padding='valid')(main_path)
        main_path = BatchNormalization()(main_path)
        main_path = Flatten()(main_path)
        main_path = Dense(units=128, activation='relu')(main_path)
        main_path = Dense(units=64, activation='relu')(main_path)
        main_path = Dense(units=10, activation='softmax')(main_path)

    # Branch path
    branch_path = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_layer)
    branch_path = MaxPooling2D(pool_size=(2, 2), strides=2, padding='valid')(branch_path)
    branch_path = BatchNormalization()(branch_path)
    branch_path = Flatten()(branch_path)
    branch_path = Dense(units=128, activation='relu')(branch_path)
    branch_path = Dense(units=64, activation='relu')(branch_path)
    branch_path = Dense(units=10, activation='softmax')(branch_path)

    # Fuse features
    fused_features = Concatenate()([main_path, branch_path])
    fused_features = Flatten()(fused_features)
    fused_features = Dense(units=128, activation='relu')(fused_features)
    fused_features = Dense(units=64, activation='relu')(fused_features)
    fused_features = Dense(units=10, activation='softmax')(fused_features)

    # Output layer
    output_layer = fused_features

    # Create model
    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model