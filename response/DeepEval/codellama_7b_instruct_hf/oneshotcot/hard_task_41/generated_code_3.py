import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, BatchNormalization, Flatten, Dense
from keras.applications import VGG16

def dl_model():
    input_layer = Input(shape=(28, 28, 1))

    # Block 1
    # Path 1
    block1_path1 = Conv2D(32, (3, 3), strides=(2, 2), padding='same', activation='relu')(input_layer)
    block1_path1 = MaxPooling2D((2, 2), strides=(2, 2), padding='same')(block1_path1)
    # Path 2
    block1_path2 = Conv2D(32, (3, 3), strides=(2, 2), padding='same', activation='relu')(input_layer)
    block1_path2 = MaxPooling2D((2, 2), strides=(2, 2), padding='same')(block1_path2)
    # Path 3
    block1_path3 = Conv2D(32, (3, 3), strides=(2, 2), padding='same', activation='relu')(input_layer)
    block1_path3 = MaxPooling2D((2, 2), strides=(2, 2), padding='same')(block1_path3)
    # Path 4
    block1_path4 = MaxPooling2D((2, 2), strides=(2, 2), padding='same')(input_layer)
    block1_path4 = Conv2D(32, (3, 3), strides=(2, 2), padding='same', activation='relu')(block1_path4)
    # Block 2
    block2_input = Concatenate()([block1_path1, block1_path2, block1_path3, block1_path4])
    block2_input = BatchNormalization()(block2_input)
    block2_input = Flatten()(block2_input)
    block2_input = Dense(128, activation='relu')(block2_input)
    block2_input = Dense(64, activation='relu')(block2_input)
    output_layer = Dense(10, activation='softmax')(block2_input)

    model = keras.Model(inputs=input_layer, outputs=output_layer)
    return model