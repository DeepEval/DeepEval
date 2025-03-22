import keras
from keras.layers import Input, Conv2D, GlobalAveragePooling2D, GlobalMaxPooling2D, Concatenate, Add, Dense, Flatten

def dl_model():
    # Input layer
    input_layer = Input(shape=(32, 32, 3))

    # Block 1: Parallel paths
    path1 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_layer)
    path2 = GlobalAveragePooling2D()(input_layer)
    path3 = Dense(units=128, activation='relu')(path2)
    path4 = Dense(units=10, activation='softmax')(path3)

    path5 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_layer)
    path6 = GlobalMaxPooling2D()(input_layer)
    path7 = Dense(units=128, activation='relu')(path6)
    path8 = Dense(units=10, activation='softmax')(path7)

    # Block 2: Channel attention
    block_output = Concatenate()([path1, path5])
    attention_weights = Dense(units=10, activation='sigmoid')(block_output)
    attention_output = Flatten()(attention_weights)
    attention_output = Reshape((1, 1, 10))(attention_output)
    attention_output = Multiply()([attention_output, block_output])

    # Block 3: Spatial attention
    block_output = Concatenate()([path2, path6])
    attention_weights = Dense(units=10, activation='sigmoid')(block_output)
    attention_output = Flatten()(attention_weights)
    attention_output = Reshape((1, 1, 10))(attention_output)
    attention_output = Multiply()([attention_output, block_output])

    # Final classification
    final_output = Add()([attention_output, block_output])
    final_output = Activation('relu')(final_output)
    final_output = Flatten()(final_output)
    final_output = Dense(units=10, activation='softmax')(final_output)

    model = keras.Model(inputs=input_layer, outputs=final_output)

    return model