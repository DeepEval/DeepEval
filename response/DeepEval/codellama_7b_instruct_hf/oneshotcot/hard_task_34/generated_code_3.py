import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, BatchNormalization, Flatten, Dense

def dl_model():
    # Input layer
    input_layer = Input(shape=(28, 28, 1))

    # Main path
    main_path = Conv2D(32, (3, 3), activation='relu')(input_layer)
    main_path = MaxPooling2D(pool_size=(2, 2))(main_path)
    main_path = Conv2D(64, (3, 3), activation='relu')(main_path)
    main_path = Concatenate()([main_path, main_path, main_path, main_path])
    main_path = BatchNormalization()(main_path)

    # Branch path
    branch_path = Conv2D(64, (3, 3), activation='relu')(input_layer)
    branch_path = MaxPooling2D(pool_size=(2, 2))(branch_path)

    # Fusion
    fusion_path = Concatenate()([main_path, branch_path])
    fusion_path = Dense(128, activation='relu')(fusion_path)
    fusion_path = Dense(10, activation='softmax')(fusion_path)

    # Model
    model = keras.Model(inputs=input_layer, outputs=fusion_path)

    return model