import keras
from keras.layers import Input, Conv2D, MaxPooling2D, GlobalAveragePooling2D, Dense, Flatten, Concatenate, Multiply

def dl_model():
    # Input layer
    input_layer = Input(shape=(32, 32, 3))

    # Branch 1
    branch1_conv = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_layer)
    branch1_pool = MaxPooling2D(pool_size=(2, 2), strides=2, padding='valid')(branch1_conv)
    branch1_gap = GlobalAveragePooling2D()(branch1_pool)
    branch1_fc1 = Dense(units=64, activation='relu')(branch1_gap)
    branch1_fc2 = Dense(units=32, activation='relu')(branch1_fc1)
    branch1_reshape = keras.layers.Reshape((32, 32, 3))(branch1_fc2)

    # Branch 2
    branch2_conv = Conv2D(filters=32, kernel_size=(5, 5), strides=(1, 1), padding='same', activation='relu')(input_layer)
    branch2_pool = MaxPooling2D(pool_size=(2, 2), strides=2, padding='valid')(branch2_conv)
    branch2_gap = GlobalAveragePooling2D()(branch2_pool)
    branch2_fc1 = Dense(units=64, activation='relu')(branch2_gap)
    branch2_fc2 = Dense(units=32, activation='relu')(branch2_fc1)
    branch2_reshape = keras.layers.Reshape((32, 32, 3))(branch2_fc2)

    # Concatenation and element-wise multiplication
    concat = Concatenate()([branch1_reshape, branch2_reshape])
    multiply = Multiply()([concat, input_layer])

    # Additional layers
    flatten_layer = Flatten()(multiply)
    dense1 = Dense(units=256, activation='relu')(flatten_layer)
    dense2 = Dense(units=128, activation='relu')(dense1)
    output_layer = Dense(units=10, activation='softmax')(dense2)

    # Model creation
    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model