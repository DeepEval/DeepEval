import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, BatchNormalization, Lambda, Flatten, Dense, AveragePooling2D

def dl_model():     

    input_layer = Input(shape=(32, 32, 3))
    
    # Block 1
    x = Lambda(lambda x: tf.split(x, num_or_size_splits=3, axis=3))(input_layer)
    
    # Three parallel paths
    path1 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(x[0])
    path2 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(x[1])
    path3 = Conv2D(filters=64, kernel_size=(5, 5), strides=(1, 1), padding='same', activation='relu')(x[2])
    
    # Concatenate the outputs
    block1_output = Concatenate(axis=3)([path1, path2, path3])
    block1_output = BatchNormalization()(block1_output)
    
    # Block 2
    x = block1_output

    # Four parallel paths
    path1 = Conv2D(filters=128, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(x)
    path2 = AveragePooling2D(pool_size=(3, 3), strides=2, padding='same')(x)
    path2 = Conv2D(filters=128, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(path2)
    path3 = Conv2D(filters=128, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(x)
    path3 = Concatenate(axis=3)([Conv2D(filters=128, kernel_size=(1, 3), strides=(1, 1), padding='same', activation='relu')(path3),
                                  Conv2D(filters=128, kernel_size=(3, 1), strides=(1, 1), padding='same', activation='relu')(path3)])
    path4 = Conv2D(filters=128, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(x)
    path4 = Conv2D(filters=128, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(path4)
    path4 = Concatenate(axis=3)([Conv2D(filters=128, kernel_size=(1, 3), strides=(1, 1), padding='same', activation='relu')(path4),
                                  Conv2D(filters=128, kernel_size=(3, 1), strides=(1, 1), padding='same', activation='relu')(path4)])
    
    # Concatenate the outputs of the four paths
    block2_output = Concatenate(axis=3)([path1, path2, path3, path4])

    # Final layers
    flatten_layer = Flatten()(block2_output)
    output_layer = Dense(units=10, activation='softmax')(flatten_layer)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model