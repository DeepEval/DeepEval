import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, BatchNormalization, Flatten, Dense, Lambda, Add
from tensorflow.keras.models import Model

def dl_model():
    input_layer = Input(shape=(32, 32, 3))

    # Main path
    main_path = []
    for kernel_size in [(1, 1), (3, 3), (5, 5)]:
        conv = Conv2D(filters=32, kernel_size=kernel_size, padding='same', activation='relu')(input_layer)
        main_path.append(conv)
    
    main_output = Concatenate()(main_path)

    # Branch path
    branch_output = Conv2D(filters=32, kernel_size=(1, 1), padding='same', activation='relu')(input_layer)

    # Addition of main and branch paths
    combined_output = Add()([main_output, branch_output])

    # Flatten and fully connected layers
    flatten_layer = Flatten()(combined_output)
    dense1 = Dense(units=128, activation='relu')(flatten_layer)
    output_layer = Dense(units=10, activation='softmax')(dense1)

    model = Model(inputs=input_layer, outputs=output_layer)
    return model

# Instantiate the model
model = dl_model()
model.summary()