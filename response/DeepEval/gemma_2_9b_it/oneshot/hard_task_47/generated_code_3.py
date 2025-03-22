import keras
from keras.layers import Input, Conv2D, Lambda, BatchNormalization, Concatenate, AveragePooling2D, Flatten, Dense

def dl_model():  

    input_layer = Input(shape=(32, 32, 3))
    
    # First Block
    split_tensor = Lambda(lambda x: tf.split(x, 3, axis=3))(input_layer)
    
    branch1 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(split_tensor[0])
    branch1 = BatchNormalization()(branch1)
    branch2 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(split_tensor[1])
    branch2 = BatchNormalization()(branch2)
    branch3 = Conv2D(filters=64, kernel_size=(5, 5), strides=(1, 1), padding='same', activation='relu')(split_tensor[2])
    branch3 = BatchNormalization()(branch3)
    
    concatenated_output = Concatenate()([branch1, branch2, branch3])

    # Second Block
    branch4 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(concatenated_output)
    branch4 = BatchNormalization()(branch4)
    branch4 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(branch4)

    branch5 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(concatenated_output)
    branch5 = BatchNormalization()(branch5)
    branch5 = Conv2D(filters=64, kernel_size=(1, 7), strides=(1, 1), padding='same', activation='relu')(branch5)
    branch5 = BatchNormalization()(branch5)
    branch5 = Conv2D(filters=64, kernel_size=(7, 1), strides=(1, 1), padding='same', activation='relu')(branch5)
    branch5 = BatchNormalization()(branch5)
    branch5 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(branch5)

    branch6 = AveragePooling2D(pool_size=(2, 2), strides=2, padding='same')(concatenated_output)

    concatenated_output = Concatenate()([branch4, branch5, branch6])
    
    flatten_layer = Flatten()(concatenated_output)
    dense1 = Dense(units=128, activation='relu')(flatten_layer)
    output_layer = Dense(units=10, activation='softmax')(dense1)
    
    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model