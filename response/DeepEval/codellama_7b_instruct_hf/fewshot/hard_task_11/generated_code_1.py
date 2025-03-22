import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense

def dl_model():
    # Define the input layer
    input_layer = Input(shape=(32, 32, 3))

    # Define the main pathway
    main_path = Conv2D(filters=16, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_layer)
    main_path = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(main_path)
    main_path = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(main_path)
    main_path = Flatten()(main_path)

    # Define the branch pathway
    branch_path = Conv2D(filters=16, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_layer)
    branch_path = Conv2D(filters=32, kernel_size=(1, 3), strides=(1, 1), padding='same', activation='relu')(branch_path)
    branch_path = Conv2D(filters=64, kernel_size=(3, 1), strides=(1, 1), padding='same', activation='relu')(branch_path)
    branch_path = Flatten()(branch_path)

    # Merge the main and branch paths
    merged_path = Add()([main_path, branch_path])

    # Define the output layer
    output_layer = Dense(units=10, activation='softmax')(merged_path)

    # Create the model
    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model