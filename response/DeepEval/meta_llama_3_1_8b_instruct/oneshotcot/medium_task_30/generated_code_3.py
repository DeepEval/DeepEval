import keras
from keras.layers import Input, AveragePooling2D, Concatenate, Flatten, Dense

def dl_model():
    # Step 1: add input layer
    input_layer = Input(shape=(32, 32, 3))

    # Step 2: add pooling layer
    pool1 = AveragePooling2D(pool_size=(1, 1), strides=(1, 1), padding='same')(input_layer)

    # Step 3: add pooling layer
    pool2 = AveragePooling2D(pool_size=(2, 2), strides=2, padding='same')(pool1)

    # Step 4: add pooling layer
    pool3 = AveragePooling2D(pool_size=(4, 4), strides=4, padding='same')(pool2)

    # Step 5: concatenate the outputs of these pooling layers
    output_tensor = Concatenate()([pool1, pool2, pool3])

    # Step 6: flatten the result
    flatten_layer = Flatten()(output_tensor)

    # Step 7: add dense layer
    dense1 = Dense(units=128, activation='relu')(flatten_layer)

    # Step 8: add dense layer
    output_layer = Dense(units=10, activation='softmax')(dense1)

    # Construct the model
    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model