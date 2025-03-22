import keras
from keras.layers import Input, Conv2D, AveragePooling2D, Lambda, Concatenate, BatchNormalization, Flatten, Dense

def dl_model():     
    input_layer = Input(shape=(32, 32, 3))

    # Block 1
    x = Lambda(lambda x: tf.split(x, 3, axis=3))(input_layer)
    
    # Branch 1: 1x1 conv
    branch1 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(x[0])
    branch1 = BatchNormalization()(branch1)
    
    # Branch 2: 3x3 conv
    branch2 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(x[1])
    branch2 = BatchNormalization()(branch2)
    
    # Branch 3: 5x5 conv
    branch3 = Conv2D(filters=64, kernel_size=(5, 5), strides=(1, 1), padding='same', activation='relu')(x[2])
    branch3 = BatchNormalization()(branch3)
    
    # Concatenate branches
    block1_output = Concatenate()([branch1, branch2, branch3])

    # Block 2
    x = block1_output
    
    # Path 1: 1x1 conv
    path1 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(x)
    
    # Path 2: 3x3 avg pool + 1x1 conv
    path2 = AveragePooling2D(pool_size=(3, 3), strides=2, padding='same')(x)
    path2 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(path2)
    
    # Path 3: 1x1 conv + split + concatenate
    path3 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(x)
    path3_1 = Conv2D(filters=64, kernel_size=(1, 3), strides=(1, 1), padding='same', activation='relu')(path3)
    path3_2 = Conv2D(filters=64, kernel_size=(3, 1), strides=(1, 1), padding='same', activation='relu')(path3)
    path3 = Concatenate()([path3_1, path3_2])
    
    # Path 4: 1x1 conv + 3x3 conv + split + concatenate
    path4 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(x)
    path4 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(path4)
    path4_1 = Conv2D(filters=64, kernel_size=(1, 3), strides=(1, 1), padding='same', activation='relu')(path4)
    path4_2 = Conv2D(filters=64, kernel_size=(3, 1), strides=(1, 1), padding='same', activation='relu')(path4)
    path4 = Concatenate()([path4_1, path4_2])

    # Concatenate all paths
    block2_output = Concatenate()([path1, path2, path3, path4])

    # Flatten and output
    flatten_layer = Flatten()(block2_output)
    output_layer = Dense(units=10, activation='softmax')(flatten_layer)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model