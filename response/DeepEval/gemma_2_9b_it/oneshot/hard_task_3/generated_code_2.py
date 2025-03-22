import tensorflow as tf
from tensorflow import keras
from keras.layers import Input, Lambda, Conv2D, Dropout, Concatenate, Flatten, Dense

def dl_model(): 
    input_tensor = Input(shape=(32, 32, 3))

    # Split input into three groups
    split_tensor = Lambda(lambda x: tf.split(x, num_or_size_splits=3, axis=2))(input_tensor)

    # Create three parallel branches
    branch1 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(split_tensor[0])
    branch1 = Conv2D(filters=128, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(branch1)
    branch1 = Dropout(rate=0.25)(branch1)

    branch2 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(split_tensor[1])
    branch2 = Conv2D(filters=128, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(branch2)
    branch2 = Dropout(rate=0.25)(branch2)

    branch3 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(split_tensor[2])
    branch3 = Conv2D(filters=128, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(branch3)
    branch3 = Dropout(rate=0.25)(branch3)

    # Concatenate outputs of the three branches
    main_pathway = Concatenate()([branch1, branch2, branch3])

    # Parallel branch
    branch_parallel = Conv2D(filters=128, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_tensor)

    # Combine outputs from main pathway and parallel branch
    combined_output = tf.keras.layers.Add()([main_pathway, branch_parallel])

    # Flatten and fully connect
    flatten_layer = Flatten()(combined_output)
    output_layer = Dense(units=10, activation='softmax')(flatten_layer)

    model = keras.Model(inputs=input_tensor, outputs=output_layer)

    return model