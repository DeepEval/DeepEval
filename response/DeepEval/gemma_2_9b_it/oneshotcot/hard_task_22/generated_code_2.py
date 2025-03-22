import keras
from keras.layers import Input, Conv2D, Lambda, Concatenate, BatchNormalization, Flatten, Dense, tf

def dl_model():
    
    input_layer = Input(shape=(32, 32, 3))

    # Main path
    x = Lambda(lambda x: tf.split(x, 3, axis=2))(input_layer)

    # Multi-scale feature extraction for each channel group
    x1 = Conv2D(filters=64, kernel_size=(1, 1), activation='relu')(x[0])
    x1 = Conv2D(filters=64, kernel_size=(3, 3), activation='relu')(x1)
    x1 = Conv2D(filters=64, kernel_size=(5, 5), activation='relu')(x1)
    x2 = Conv2D(filters=64, kernel_size=(1, 1), activation='relu')(x[1])
    x2 = Conv2D(filters=64, kernel_size=(3, 3), activation='relu')(x2)
    x2 = Conv2D(filters=64, kernel_size=(5, 5), activation='relu')(x2)
    x3 = Conv2D(filters=64, kernel_size=(1, 1), activation='relu')(x[2])
    x3 = Conv2D(filters=64, kernel_size=(3, 3), activation='relu')(x3)
    x3 = Conv2D(filters=64, kernel_size=(5, 5), activation='relu')(x3)

    # Concatenate outputs from each channel group
    main_output = Concatenate(axis=2)([x1, x2, x3])

    # Branch path
    branch_output = Conv2D(filters=192, kernel_size=(1, 1), activation='relu')(input_layer)

    # Fuse outputs from main and branch paths
    output = main_output + branch_output

    # Flatten and fully connected layers
    output = Flatten()(output)
    output = Dense(units=128, activation='relu')(output)
    output = Dense(units=10, activation='softmax')(output)

    model = keras.Model(inputs=input_layer, outputs=output)
    return model