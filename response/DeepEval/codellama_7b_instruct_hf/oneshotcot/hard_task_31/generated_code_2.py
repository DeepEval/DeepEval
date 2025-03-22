import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Dropout, BatchNormalization, Flatten, Dense, Lambda

def dl_model():
    # Define the input layer
    input_layer = Input(shape=(32, 32, 3))

    # First block
    main_path = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_layer)
    main_path = Dropout(rate=0.2)(main_path)
    main_path = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(main_path)
    main_path = Dropout(rate=0.2)(main_path)

    branch_path = input_layer

    # Add the main and branch path
    output_tensor = main_path + branch_path

    # Batch normalization
    output_tensor = BatchNormalization()(output_tensor)

    # Flatten
    output_tensor = Flatten()(output_tensor)

    # Fully connected layers
    output_tensor = Dense(units=128, activation='relu')(output_tensor)
    output_tensor = Dense(units=10, activation='softmax')(output_tensor)

    # Define the model
    model = keras.Model(inputs=input_layer, outputs=output_tensor)

    return model