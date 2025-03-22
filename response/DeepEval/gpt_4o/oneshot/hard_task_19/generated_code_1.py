import keras
from keras.layers import Input, Conv2D, MaxPooling2D, GlobalAveragePooling2D, Dense, Multiply, Add, Flatten

def dl_model():
    input_layer = Input(shape=(32, 32, 3))

    # Main Path
    main_path = Conv2D(filters=32, kernel_size=(3, 3), padding='same', activation='relu')(input_layer)
    main_path = Conv2D(filters=64, kernel_size=(3, 3), padding='same', activation='relu')(main_path)
    main_path = Conv2D(filters=128, kernel_size=(3, 3), padding='same', activation='relu')(main_path)
    main_path = MaxPooling2D(pool_size=(2, 2), padding='same')(main_path)
    
    # Branch Path for Channel Weighting
    branch_path = GlobalAveragePooling2D()(main_path)
    branch_path = Dense(units=64, activation='relu')(branch_path)
    branch_path = Dense(units=128, activation='sigmoid')(branch_path)  # channel weights
    branch_path = Multiply()([main_path, branch_path])  # Reshape and multiply with main path

    # Combine Paths
    combined = Add()([main_path, branch_path])

    # Fully Connected Layers for Classification
    fc1 = Flatten()(combined)
    fc1 = Dense(units=128, activation='relu')(fc1)
    fc2 = Dense(units=64, activation='relu')(fc1)
    output_layer = Dense(units=10, activation='softmax')(fc2)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model