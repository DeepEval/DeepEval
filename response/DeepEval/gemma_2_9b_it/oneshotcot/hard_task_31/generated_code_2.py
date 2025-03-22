import keras
from keras.layers import Input, Conv2D, Dropout, Lambda, Concatenate, BatchNormalization, Flatten, Dense, tf
 
def dl_model():  

    input_layer = Input(shape=(32, 32, 3))
 
    # Block 1
    x_main = Conv2D(filters=64, kernel_size=(3, 3), padding='same', activation='relu')(input_layer)
    x_main = Dropout(0.2)(x_main) 
    x_main = Conv2D(filters=32, kernel_size=(3, 3), padding='same', activation='relu')(x_main) 
    x_branch = input_layer 
    x_block1 = keras.layers.add([x_main, x_branch])

    # Block 2
    def split_and_process(inputs):
        split_tensor = tf.split(inputs, num_or_size_splits=3, axis=-1)
        conv1 = Conv2D(filters=64, kernel_size=(1, 1), padding='same', activation='relu')(split_tensor[0])
        conv1 = Dropout(0.2)(conv1)
        conv2 = Conv2D(filters=64, kernel_size=(3, 3), padding='same', activation='relu')(split_tensor[1])
        conv2 = Dropout(0.2)(conv2)
        conv3 = Conv2D(filters=64, kernel_size=(5, 5), padding='same', activation='relu')(split_tensor[2])
        conv3 = Dropout(0.2)(conv3)
        return Concatenate()([conv1, conv2, conv3])

    x_block2 = Lambda(split_and_process)(x_block1)
    
    x = Flatten()(x_block2)
    output_layer = Dense(units=10, activation='softmax')(x) 

    model = keras.Model(inputs=input_layer, outputs=output_layer) 
    return model