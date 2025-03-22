import keras
from keras.layers import Input, Conv2D, AveragePooling2D, Lambda, Concatenate, BatchNormalization, Flatten, Dense
from keras.layers import tf

def dl_model(): 
    
    input_layer = Input(shape=(32, 32, 3)) 

    # Block 1
    x = Lambda(lambda x: tf.split(x, 3, axis=3))(input_layer)
    
    # Group 1
    branch1 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(x[0])
    branch1 = BatchNormalization()(branch1) 
    
    # Group 2
    branch2 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(x[1])
    branch2 = BatchNormalization()(branch2) 
    
    # Group 3
    branch3 = Conv2D(filters=64, kernel_size=(5, 5), strides=(1, 1), padding='same', activation='relu')(x[2])
    branch3 = BatchNormalization()(branch3)

    # Concatenate outputs of the three groups
    x = Concatenate(axis=3)([branch1, branch2, branch3])

    # Block 2
    x = Lambda(lambda x: tf.split(x, 4, axis=3))(x)
    
    # Path 1
    path1 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(x[0])
    
    # Path 2
    path2 = AveragePooling2D(pool_size=(3, 3), strides=(2, 2), padding='same')(x[1])
    path2 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(path2)

    # Path 3
    path3 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(x[2])
    path3 = tf.split(path3, 2, axis=3)
    path3 = [Conv2D(filters=64, kernel_size=(1, 3), strides=(1, 1), padding='same', activation='relu')(p) for p in path3]
    path3 = Concatenate(axis=3)(path3)

    # Path 4
    path4 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(x[3])
    path4 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(path4)
    path4 = tf.split(path4, 2, axis=3)
    path4 = [Conv2D(filters=64, kernel_size=(3, 1), strides=(1, 1), padding='same', activation='relu')(p) for p in path4]
    path4 = Concatenate(axis=3)(path4)

    # Concatenate outputs of all four paths
    x = Concatenate(axis=3)([path1, path2, path3, path4])

    x = Flatten()(x)
    output_layer = Dense(units=10, activation='softmax')(x)
    model = keras.Model(inputs=input_layer, outputs=output_layer)
    
    return model