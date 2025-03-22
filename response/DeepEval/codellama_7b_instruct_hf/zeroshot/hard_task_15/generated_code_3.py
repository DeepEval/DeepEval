import keras
from keras.layers import Input, GlobalAveragePooling2D, Dense, Conv2D, Flatten, Add

def dl_model():
    # Main path
    main_path = keras.Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)),
        GlobalAveragePooling2D(),
        Dense(64, activation='relu'),
        Dense(10)
    ])

    # Branch path
    branch_path = keras.Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)),
        Flatten(),
        Dense(64, activation='relu'),
        Dense(10)
    ])

    # Combine main and branch paths
    main_path = main_path(branch_path.output)
    main_path = Add()([main_path, branch_path])
    main_path = Dense(10)(main_path)

    return keras.Model(inputs=branch_path.input, outputs=main_path)