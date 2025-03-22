import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, BatchNormalization, Flatten, Dense

def dl_model():
    # Define the input layer
    input_layer = Input(shape=(32, 32, 3))

    # Define the main pathway
    main_pathway = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_layer)
    parallel_pathway = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_layer)
    parallel_pathway = Conv2D(filters=64, kernel_size=(1, 3), strides=(1, 1), padding='same', activation='relu')(parallel_pathway)
    parallel_pathway = Conv2D(filters=64, kernel_size=(3, 1), strides=(1, 1), padding='same', activation='relu')(parallel_pathway)
    main_pathway = Concatenate()([main_pathway, parallel_pathway])
    main_pathway = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(main_pathway)

    # Define the output layer
    output_layer = Dense(units=10, activation='softmax')(main_pathway)

    # Define the model
    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model