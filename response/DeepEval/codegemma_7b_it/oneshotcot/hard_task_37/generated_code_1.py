import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Add, Concatenate, BatchNormalization, Flatten, Dense

def dl_model():

    # Input layer
    input_layer = Input(shape=(28, 28, 1))

    # Parallel branch 1
    branch1 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='valid', activation='relu')(input_layer)
    branch1 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='valid', activation='relu')(branch1)
    branch1 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='valid', activation='relu')(branch1)

    # Parallel branch 2
    branch2 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='valid', activation='relu')(input_layer)
    branch2 = Conv2D(filters=32, kernel_size=(5, 5), strides=(1, 1), padding='valid', activation='relu')(branch2)
    branch2 = Conv2D(filters=32, kernel_size=(7, 7), strides=(1, 1), padding='valid', activation='relu')(branch2)

    # Parallel branch 3
    branch3 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='valid', activation='relu')(input_layer)
    branch3 = MaxPooling2D(pool_size=(2, 2), strides=2, padding='valid')(branch3)
    branch3 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='valid', activation='relu')(branch3)
    branch3 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='valid', activation='relu')(branch3)

    # Concatenate branches 1 and 2
    concat_branch1_2 = Concatenate()([branch1, branch2])

    # Concatenate branch 3
    concat_branch1_2_3 = Concatenate()([concat_branch1_2, branch3])

    # Batch normalization
    batch_norm = BatchNormalization()(concat_branch1_2_3)

    # Flatten layer
    flatten_layer = Flatten()(batch_norm)

    # Fully connected layer
    dense = Dense(units=128, activation='relu')(flatten_layer)

    # Output layer
    output_layer = Dense(units=10, activation='softmax')(dense)

    # Model definition
    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model