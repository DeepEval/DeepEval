import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, BatchNormalization, Flatten, Dense, Lambda, tf

def dl_model():
    
    input_layer = Input(shape=(32, 32, 3))

    # Main path
    x = Lambda(lambda x: tf.split(x, num_or_size_splits=3, axis=2))(input_layer)
    
    # 1x1 conv path
    x1 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(x[0])
    
    # 3x3 conv path
    x2 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(x[1])
    
    # 5x5 conv path
    x3 = Conv2D(filters=64, kernel_size=(5, 5), strides=(1, 1), padding='same', activation='relu')(x[2])

    # Concatenate outputs from main path
    main_path_output = Concatenate(axis=2)([x1, x2, x3])
    
    # Branch path
    branch_path = Conv2D(filters=192, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_layer)

    # Add main path and branch path outputs
    output = keras.layers.Add()([main_path_output, branch_path])

    # Flatten and dense layers
    x = Flatten()(output)
    x = Dense(units=128, activation='relu')(x)
    output_layer = Dense(units=10, activation='softmax')(x)

    model = keras.Model(inputs=input_layer, outputs=output_layer)
    
    return model