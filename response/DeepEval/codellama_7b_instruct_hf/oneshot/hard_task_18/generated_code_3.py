import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, BatchNormalization, Flatten, Dense

def dl_model():
    # Define the input layer
    input_layer = Input(shape=(32, 32, 3))

    # First sequential block
    conv1 = Conv2D(32, (3, 3), activation='relu')(input_layer)
    conv2 = Conv2D(32, (3, 3), activation='relu')(conv1)
    max_pooling = MaxPooling2D((2, 2))(conv2)
    block1 = Concatenate()([input_layer, max_pooling])

    # Second sequential block
    block1_output = block1
    main_path = BatchNormalization()(block1_output)
    main_path_output = Flatten()(main_path)
    dense1 = Dense(64, activation='relu')(main_path_output)
    dense2 = Dense(10, activation='softmax')(dense1)
    output_layer = Dense(10, activation='softmax')(dense2)

    # Define the model
    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model