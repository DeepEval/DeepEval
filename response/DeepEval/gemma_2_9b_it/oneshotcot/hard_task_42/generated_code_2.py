import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dropout, Concatenate, BatchNormalization, Dense, Reshape

def dl_model():
    
    input_layer = Input(shape=(28, 28, 1))

    # Block 1
    path1 = MaxPooling2D(pool_size=(1, 1), strides=(1, 1))(input_layer)
    path2 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(input_layer)
    path3 = MaxPooling2D(pool_size=(4, 4), strides=(4, 4))(input_layer)

    flat1 = Flatten()(path1)
    flat2 = Flatten()(path2)
    flat3 = Flatten()(path3)

    drop1 = Dropout(0.2)(flat1)
    drop2 = Dropout(0.2)(flat2)
    drop3 = Dropout(0.2)(flat3)

    block1_output = Concatenate()([drop1, drop2, drop3])

    # Fully Connected and Reshape
    fc1 = Dense(units=128, activation='relu')(block1_output)
    reshape_layer = Reshape((1, 128))(fc1)

    # Block 2
    path1 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(reshape_layer)
    path2 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(reshape_layer)
    path2 = Conv2D(filters=64, kernel_size=(7, 1), strides=(1, 1), padding='same', activation='relu')(path2)
    path2 = Conv2D(filters=64, kernel_size=(1, 7), strides=(1, 1), padding='same', activation='relu')(path2)
    path3 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(reshape_layer)
    path3 = Conv2D(filters=64, kernel_size=(7, 1), strides=(1, 1), padding='same', activation='relu')(path3)
    path3 = Conv2D(filters=64, kernel_size=(1, 7), strides=(1, 1), padding='same', activation='relu')(path3)
    path4 = AveragePooling2D(pool_size=(2, 2), strides=(2, 2))(reshape_layer)
    path4 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(path4)

    block2_output = Concatenate(axis=1)([path1, path2, path3, path4])

    # Final Layers
    flatten = Flatten()(block2_output)
    dense2 = Dense(units=64, activation='relu')(flatten)
    output_layer = Dense(units=10, activation='softmax')(dense2)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model