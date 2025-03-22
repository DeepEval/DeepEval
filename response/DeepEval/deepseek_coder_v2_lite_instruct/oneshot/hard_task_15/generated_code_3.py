import keras
from keras.layers import Input, Conv2D, GlobalAveragePooling2D, Dense, Add, Multiply
from keras.models import Model

def dl_model():
    input_layer = Input(shape=(32, 32, 3))

    # Main path
    main_path = Conv2D(filters=32, kernel_size=(3, 3), activation='relu')(input_layer)
    main_path = GlobalAveragePooling2D()(main_path)
    main_path = Dense(units=64, activation='relu')(main_path)
    weights = Dense(units=32, activation='relu')(main_path)
    weights = weights.reshape((1, 1, 32))  # Reshape to match input shape
    main_path = Multiply()([input_layer, weights])  # Element-wise multiplication

    # Branch path
    branch_path = input_layer

    # Combine both paths
    combined = Add()([main_path, branch_path])

    # Additional fully connected layers
    combined = Dense(units=128, activation='relu')(combined)
    output_layer = Dense(units=10, activation='softmax')(combined)

    model = Model(inputs=input_layer, outputs=output_layer)

    return model