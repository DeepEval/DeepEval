import numpy as np
from keras.models import Model
from keras.layers import Input, GlobalAveragePooling2D, Dense, Flatten


def dl_model():
    # Input layer
    input_layer = Input(shape=(32, 32, 3))

    # Main path
    main_path = GlobalAveragePooling2D()(input_layer)
    main_path = Dense(64, activation='relu')(main_path)
    main_path = Dense(10, activation='softmax')(main_path)

    # Branch path
    branch_path = Dense(64, activation='relu')(input_layer)
    branch_path = Dense(10, activation='softmax')(branch_path)

    # Combine main and branch paths
    combined_path = Add()([main_path, branch_path])

    # Final layers
    flattened_path = Flatten()(combined_path)
    output_layer = Dense(10, activation='softmax')(flattened_path)

    # Define the model
    model = Model(inputs=input_layer, outputs=output_layer)

    # Compile the model
    model.compile(optimizer='adam',
                loss='categorical_crossentropy',
                metrics=['accuracy'])

    # Return the model
    return model