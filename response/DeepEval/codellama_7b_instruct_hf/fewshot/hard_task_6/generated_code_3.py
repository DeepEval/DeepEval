import keras
from keras.layers import Input, Conv2D, MaxPooling2D, AveragePooling2D, Lambda, Concatenate, Reshape, Dense

def dl_model():
    input_layer = Input(shape=(32, 32, 3))

    # Main path
    # Block 1
    blocks_1 = []
    for i in range(3):
        blocks_1.append(Conv2D(32, kernel_size=1, activation='relu')(input_layer))
    main_path = Concatenate()(blocks_1)

    # Block 2
    main_path = Reshape(target_shape=(32, 32, 3))(main_path)
    main_path = Permute((3, 1, 2))(main_path)
    main_path = Reshape(target_shape=(32, 32, 3))(main_path)

    # Block 3
    main_path = Conv2D(64, kernel_size=3, activation='relu')(main_path)
    main_path = MaxPooling2D(pool_size=(2, 2))(main_path)

    # Block 4 (repeated)
    main_path = Reshape(target_shape=(32, 32, 3))(main_path)
    main_path = Permute((3, 1, 2))(main_path)
    main_path = Reshape(target_shape=(32, 32, 3))(main_path)
    main_path = Conv2D(128, kernel_size=3, activation='relu')(main_path)
    main_path = MaxPooling2D(pool_size=(2, 2))(main_path)

    # Branch path
    branch_path = AveragePooling2D(pool_size=(2, 2))(input_layer)

    # Combine both paths
    combined_path = Concatenate()([main_path, branch_path])

    # Fully connected layer
    output_layer = Dense(10, activation='softmax')(combined_path)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model