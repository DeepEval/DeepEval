import keras
from keras.layers import Input, Conv2D, GlobalAveragePooling2D, Concatenate, Dense, Reshape
from keras.models import Model

def dl_model():
    input_layer = Input(shape=(32, 32, 3))

    # Main path
    main_path = Conv2D(filters=32, kernel_size=(3, 3), activation='relu')(input_layer)
    main_path = Conv2D(filters=32, kernel_size=(3, 3), activation='relu')(main_path)
    main_path = GlobalAveragePooling2D()(main_path)
    main_path = Dense(units=32, activation='relu')(main_path)
    main_path = Reshape((1, 1, 32))(main_path)  # Reshape to match input shape
    main_path = Multiply()([input_layer, main_path])  # Element-wise multiplication

    # Branch path
    branch_path = Conv2D(filters=32, kernel_size=(3, 3), activation='relu')(input_layer)
    branch_path = Conv2D(filters=32, kernel_size=(3, 3), activation='relu')(branch_path)

    # Combine paths
    combined = Concatenate()([main_path, branch_path])

    # Flatten and add fully connected layers
    combined = GlobalAveragePooling2D()(combined)
    dense1 = Dense(units=128, activation='relu')(combined)
    dense2 = Dense(units=64, activation='relu')(dense1)
    output_layer = Dense(units=10, activation='softmax')(dense2)

    model = Model(inputs=input_layer, outputs=output_layer)
    return model