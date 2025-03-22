import keras
from keras.layers import Input, Conv2D, Lambda, Concatenate, BatchNormalization, Flatten, Dense
from tensorflow import split

def dl_model():
    input_layer = Input(shape=(32, 32, 3))

    # Main path
    x = Lambda(lambda x: split(x, num_or_size_splits=3, axis=2))(input_layer) 
    
    # Group 1: 1x1 conv
    x1 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(x[0])

    # Group 2: 3x3 conv
    x2 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(x[1])

    # Group 3: 5x5 conv
    x3 = Conv2D(filters=64, kernel_size=(5, 5), strides=(1, 1), padding='same', activation='relu')(x[2])

    main_path_output = Concatenate()([x1, x2, x3])

    # Branch path
    branch_path = Conv2D(filters=192, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_layer)

    # Add main and branch paths
    output = main_path_output + branch_path

    # Fully connected layers
    output = Flatten()(output)
    output = Dense(units=128, activation='relu')(output)
    output = Dense(units=10, activation='softmax')(output)

    model = keras.Model(inputs=input_layer, outputs=output)

    return model