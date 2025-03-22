import keras
from keras.layers import Input, Conv2D, Lambda, Concatenate, BatchNormalization, AveragePooling2D, Flatten, Dense

def dl_model():     

    input_layer = Input(shape=(32, 32, 3))
    
    # First Block
    x = Lambda(lambda x: tf.split(x, 3, axis=-1))(input_layer)
    
    # Processing each group
    group1 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(x[0])
    group2 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(x[1])
    group3 = Conv2D(filters=64, kernel_size=(5, 5), strides=(1, 1), padding='same', activation='relu')(x[2])
    
    group1 = BatchNormalization()(group1)
    group2 = BatchNormalization()(group2)
    group3 = BatchNormalization()(group3)

    # Concatenate outputs of groups
    x = Concatenate()([group1, group2, group3]) 

    # Second Block
    branch1 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(x)
    branch1 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(branch1)

    branch2 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(x)
    branch2 = Conv2D(filters=64, kernel_size=(1, 7), strides=(1, 1), padding='same', activation='relu')(branch2)
    branch2 = Conv2D(filters=64, kernel_size=(7, 1), strides=(1, 1), padding='same', activation='relu')(branch2)
    branch2 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(branch2)

    branch3 = AveragePooling2D(pool_size=(2, 2), strides=2, padding='valid')(x)

    x = Concatenate()([branch1, branch2, branch3])

    # Flatten and Fully Connected Layers
    x = Flatten()(x)
    x = Dense(units=128, activation='relu')(x)
    output_layer = Dense(units=10, activation='softmax')(x) 

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model