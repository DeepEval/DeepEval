import keras
from keras.layers import Input, GlobalAveragePooling2D, Dense, Multiply, Add, Flatten

def dl_model():

    input_layer = Input(shape=(32, 32, 3))

    # Main path
    gap = GlobalAveragePooling2D()(input_layer)
    dense1 = Dense(filters=32, activation='relu')(gap)
    dense2 = Dense(filters=32, activation='relu')(dense1)
    weight = Dense(filters=32, activation='sigmoid')(dense2)
    reshape_weight = Reshape((32, 32, 3))(weight)
    multiply = Multiply()([reshape_weight, input_layer])

    # Branch path
    branch_path = input_layer

    # Combining both paths
    combined = Add()([multiply, branch_path])

    # Fully connected layers
    flatten = Flatten()(combined)
    dense3 = Dense(units=64, activation='relu')(flatten)
    output_layer = Dense(units=10, activation='softmax')(dense3)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model