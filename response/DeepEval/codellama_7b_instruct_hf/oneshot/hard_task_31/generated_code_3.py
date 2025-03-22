import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, BatchNormalization, Flatten, Dense

def dl_model():
    input_layer = Input(shape=(32, 32, 3))

    # First block: Main path and branch path
    main_path = Conv2D(32, (3, 3), activation='relu')(input_layer)
    main_path = MaxPooling2D((2, 2))(main_path)
    main_path = Conv2D(64, (3, 3), activation='relu')(main_path)
    branch_path = Conv2D(64, (1, 1), activation='relu')(input_layer)
    branch_path = Conv2D(128, (3, 3), activation='relu')(branch_path)
    branch_path = MaxPooling2D((2, 2))(branch_path)
    outputs = Concatenate()([main_path, branch_path])

    # Second block: Separable convolutions with varying kernel sizes
    outputs = Conv2D(128, (1, 1), activation='relu')(outputs)
    outputs = Conv2D(128, (3, 3), activation='relu')(outputs)
    outputs = MaxPooling2D((2, 2))(outputs)
    outputs = Conv2D(128, (5, 5), activation='relu')(outputs)
    outputs = MaxPooling2D((2, 2))(outputs)

    # Flatten and fully connected layers
    outputs = Flatten()(outputs)
    outputs = Dense(128, activation='relu')(outputs)
    outputs = Dense(10, activation='softmax')(outputs)

    model = keras.Model(inputs=input_layer, outputs=outputs)
    return model