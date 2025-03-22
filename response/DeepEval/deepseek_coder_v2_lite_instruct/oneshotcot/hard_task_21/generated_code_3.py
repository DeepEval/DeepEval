import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, BatchNormalization, Flatten, Dense, Lambda, Add
from tensorflow.keras.models import Model

def dl_model():
    input_layer = Input(shape=(32, 32, 3))

    # Main path
    main_path = [
        Conv2D(32, (1, 1), padding='same', activation='relu')(input_layer),
        Conv2D(64, (3, 3), padding='same', activation='relu')(input_layer),
        Conv2D(64, (5, 5), padding='same', activation='relu')(input_layer)
    ]
    main_output = Concatenate()(main_path)

    # Branch path
    branch_path = Conv2D(32, (1, 1), padding='same', activation='relu')(input_layer)

    # Add the outputs of the main and branch paths
    added_output = Add()([main_output, branch_path])

    # Batch normalization and flattening
    batch_norm = BatchNormalization()(added_output)
    flatten_layer = Flatten()(batch_norm)

    # Fully connected layers
    dense1 = Dense(128, activation='relu')(flatten_layer)
    dense2 = Dense(64, activation='relu')(dense1)
    output_layer = Dense(10, activation='softmax')(dense2)

    model = Model(inputs=input_layer, outputs=output_layer)

    return model

# Build the model
model = dl_model()
model.summary()