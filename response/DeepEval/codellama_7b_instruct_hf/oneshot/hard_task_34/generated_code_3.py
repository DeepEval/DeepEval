import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, BatchNormalization, Flatten, Dense

def dl_model():
    input_layer = Input(shape=(28, 28, 1))

    # Main path
    main_path = Conv2D(32, (3, 3), activation='relu')(input_layer)
    main_path = MaxPooling2D((2, 2))(main_path)
    main_path = Conv2D(64, (3, 3), activation='relu')(main_path)
    main_path = Concatenate()([main_path, Conv2D(64, (3, 3), activation='relu')(main_path)])
    main_path = BatchNormalization()(main_path)
    main_path = Flatten()(main_path)
    main_path = Dense(128, activation='relu')(main_path)
    main_path = Dense(64, activation='relu')(main_path)
    main_path = Dense(10, activation='softmax')(main_path)

    # Branch path
    branch_path = Conv2D(32, (3, 3), activation='relu')(input_layer)
    branch_path = MaxPooling2D((2, 2))(branch_path)
    branch_path = Conv2D(64, (3, 3), activation='relu')(branch_path)
    branch_path = Concatenate()([branch_path, Conv2D(64, (3, 3), activation='relu')(branch_path)])
    branch_path = BatchNormalization()(branch_path)
    branch_path = Flatten()(branch_path)
    branch_path = Dense(128, activation='relu')(branch_path)
    branch_path = Dense(64, activation='relu')(branch_path)
    branch_path = Dense(10, activation='softmax')(branch_path)

    # Fusion layer
    fusion_layer = Concatenate()([main_path, branch_path])
    fusion_layer = Dense(64, activation='relu')(fusion_layer)
    fusion_layer = Dense(32, activation='relu')(fusion_layer)
    fusion_layer = Dense(10, activation='softmax')(fusion_layer)

    model = keras.Model(inputs=input_layer, outputs=fusion_layer)
    return model