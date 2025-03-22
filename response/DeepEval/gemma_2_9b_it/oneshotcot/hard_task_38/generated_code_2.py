import keras
from keras.layers import Input, Conv2D, BatchNormalization, Activation, concatenate, Flatten, Dense

def dl_model():     

    input_layer = Input(shape=(28, 28, 1))

    # Pathway 1
    path1_input = input_layer
    for _ in range(3):
      path1_input = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(path1_input)
      path1_input = BatchNormalization()(path1_input)
    
    # Pathway 2
    path2_input = input_layer
    for _ in range(3):
      path2_input = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(path2_input)
      path2_input = BatchNormalization()(path2_input)

    # Concatenate pathways
    merged = concatenate([path1_input, path2_input], axis=3)

    # Flatten and classify
    flatten_layer = Flatten()(merged)
    dense1 = Dense(units=128, activation='relu')(flatten_layer)
    output_layer = Dense(units=10, activation='softmax')(dense1)

    model = keras.Model(inputs=input_layer, outputs=output_layer)
    return model