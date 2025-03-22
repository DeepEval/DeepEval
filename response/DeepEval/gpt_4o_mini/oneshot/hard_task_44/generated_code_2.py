import keras
from keras.layers import Input, Conv2D, Dropout, Concatenate, Flatten, Dense, Lambda
import tensorflow as tf

def dl_model():
    input_layer = Input(shape=(32, 32, 3))

    # Block 1
    split_inputs = Lambda(lambda x: tf.split(x, 3, axis=-1))(input_layer)
    
    # Convolutional layers with different kernel sizes
    path1 = Conv2D(filters=32, kernel_size=(1, 1), padding='same', activation='relu')(split_inputs[0])
    path2 = Conv2D(filters=32, kernel_size=(3, 3), padding='same', activation='relu')(split_inputs[1])
    path3 = Conv2D(filters=32, kernel_size=(5, 5), padding='same', activation='relu')(split_inputs[2])
    
    # Dropout layer
    dropout_layer = Dropout(0.5)(Concatenate()([path1, path2, path3]))

    # Block 2
    branch1 = Conv2D(filters=64, kernel_size=(1, 1), padding='same', activation='relu')(dropout_layer)
    branch2 = Conv2D(filters=64, kernel_size=(1, 1), padding='same', activation='relu')(Conv2D(filters=64, kernel_size=(3, 3), padding='same', activation='relu')(dropout_layer))
    branch3 = Concatenate()([Conv2D(filters=64, kernel_size=(1, 1), padding='same', activation='relu')(dropout_layer), 
                              Conv2D(filters=64, kernel_size=(5, 5), padding='same', activation='relu')(dropout_layer)])
    branch4 = Conv2D(filters=64, kernel_size=(1, 1), padding='same', activation='relu')(Conv2D(filters=64, kernel_size=(3, 3), padding='same', activation='relu')(dropout_layer))

    # Max pooling branch
    max_pooling_branch = Conv2D(filters=64, kernel_size=(1, 1), padding='same', activation='relu')(tf.keras.layers.MaxPooling2D(pool_size=(3, 3), strides=(1, 1), padding='same')(dropout_layer))
    
    # Concatenate all branches
    block2_output = Concatenate()([branch1, branch2, branch3, max_pooling_branch])
    
    # Flatten and fully connected layers
    flatten_layer = Flatten()(block2_output)
    dense1 = Dense(units=128, activation='relu')(flatten_layer)
    output_layer = Dense(units=10, activation='softmax')(dense1)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model