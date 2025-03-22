import keras
from keras.layers import Input, Conv2D, Add, Flatten, Dense, Concatenate

def dl_model():
    input_layer = Input(shape=(32, 32, 64))

    # Main path
    conv1x1_main = Conv2D(filters=32, kernel_size=(1, 1), activation='relu')(input_layer)
    conv1x1_branch = Conv2D(filters=32, kernel_size=(1, 1), activation='relu')(input_layer)
    conv3x3_main = Conv2D(filters=32, kernel_size=(3, 3), activation='relu')(conv1x1_main)
    
    concat_main = Concatenate()([conv1x1_main, conv3x3_main])

    # Branch path
    conv3x3_branch = Conv2D(filters=32, kernel_size=(3, 3), activation='relu')(conv1x1_branch)

    # Combine paths
    adding_layer = Add()([concat_main, conv3x3_branch])

    # Classification layers
    flatten_layer = Flatten()(adding_layer)
    dense1 = Dense(units=64, activation='relu')(flatten_layer)
    output_layer = Dense(units=10, activation='softmax')(dense1)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model