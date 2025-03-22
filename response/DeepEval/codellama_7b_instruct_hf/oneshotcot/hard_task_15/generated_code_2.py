import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, GlobalAveragePooling2D, Concatenate
from keras.models import Model

def dl_model():
    # Input layer
    input_layer = Input(shape=(32, 32, 3))

    # Main path
    main_path = Conv2D(filters=32, kernel_size=(3, 3), activation='relu')(input_layer)
    main_path = MaxPooling2D(pool_size=(2, 2))(main_path)
    main_path = Flatten()(main_path)
    main_path = Dense(units=128, activation='relu')(main_path)
    main_path = Dense(units=10, activation='softmax')(main_path)

    # Branch path
    branch_path = Conv2D(filters=32, kernel_size=(3, 3), activation='relu')(input_layer)
    branch_path = MaxPooling2D(pool_size=(2, 2))(branch_path)
    branch_path = Flatten()(branch_path)
    branch_path = Dense(units=128, activation='relu')(branch_path)
    branch_path = Dense(units=10, activation='softmax')(branch_path)

    # Combine main and branch paths
    output_layer = Concatenate()([main_path, branch_path])

    # Final layers
    output_layer = Dense(units=128, activation='relu')(output_layer)
    output_layer = Dense(units=10, activation='softmax')(output_layer)

    # Create model
    model = Model(inputs=input_layer, outputs=output_layer)

    return model