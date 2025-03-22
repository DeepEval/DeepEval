import keras
from keras.layers import Input, MaxPooling2D, Flatten, Dense, Conv2D, concatenate

def dl_model():

    input_layer = Input(shape=(28, 28, 1))

    # Block 1: Max Pooling Layers
    block1_output = []
    for size in [(1, 1), (2, 2), (4, 4)]:
        x = input_layer
        for stride in size:
            x = MaxPooling2D(pool_size=(stride, stride), strides=stride, padding='valid')(x)
        block1_output.append(Flatten()(x))

    # Block 2: Feature Extraction Branches
    block2_output = []
    x = input_layer
    block2_output.append(Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(x))
    block2_output.append(Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(x))
    block2_output.append(Conv2D(filters=64, kernel_size=(5, 5), strides=(1, 1), padding='same', activation='relu')(x))
    block2_output.append(MaxPooling2D(pool_size=(3, 3), strides=2, padding='valid')(x))

    # Concatenation and Classification
    merged_output = concatenate(block1_output + block2_output)
    flatten = Flatten()(merged_output)
    dense = Dense(units=64, activation='relu')(flatten)
    output_layer = Dense(units=10, activation='softmax')(dense)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model