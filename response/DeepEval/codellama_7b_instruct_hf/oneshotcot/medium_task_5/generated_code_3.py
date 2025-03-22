import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, BatchNormalization, Flatten, Dense

def dl_model():

    input_layer = Input(shape=(32, 32, 3))

    # Main path
    main_path = Conv2D(32, (3, 3), activation='relu')(input_layer)
    main_path = Conv2D(64, (3, 3), activation='relu')(main_path)
    main_path = MaxPooling2D((2, 2))(main_path)
    main_path = Conv2D(128, (3, 3), activation='relu')(main_path)
    main_path = MaxPooling2D((2, 2))(main_path)
    main_path = Flatten()(main_path)

    # Branch path
    branch_path = Conv2D(32, (3, 3), activation='relu')(input_layer)
    branch_path = MaxPooling2D((2, 2))(branch_path)
    branch_path = Flatten()(branch_path)

    # Combine main and branch paths
    combined_path = Concatenate()([main_path, branch_path])

    # Project onto probability distribution
    output_layer = Dense(10, activation='softmax')(combined_path)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model