import keras
from keras.layers import Input, Conv2D, Lambda, MaxPooling2D, Concatenate, BatchNormalization, Flatten, Dense

def dl_model():     

    input_layer = Input(shape=(32, 32, 3))

    # Main Path
    x = Lambda(lambda x: tf.split(x, 3, axis=2))(input_layer)
    
    # Group 1: 1x1 convolutions
    group1 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(x[0])
    
    # Group 2: 3x3 convolutions
    group2 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(x[1])
    
    # Group 3: 5x5 convolutions
    group3 = Conv2D(filters=64, kernel_size=(5, 5), strides=(1, 1), padding='same', activation='relu')(x[2])

    main_path_output = Concatenate(axis=2)([group1, group2, group3])

    # Branch Path
    branch_path = Conv2D(filters=192, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_layer)

    # Fusion
    fused_output = main_path_output + branch_path

    # Flatten and Fully Connected Layers
    flatten_layer = Flatten()(fused_output)
    dense1 = Dense(units=128, activation='relu')(flatten_layer)
    output_layer = Dense(units=10, activation='softmax')(dense1)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model