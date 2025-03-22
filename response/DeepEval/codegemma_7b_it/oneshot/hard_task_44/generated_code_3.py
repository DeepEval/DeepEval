import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, BatchNormalization, Flatten, Dense, Dropout, Lambda, Reshape
from tensorflow import reshape

def dl_model():

    input_layer = Input(shape=(32, 32, 3))

    # Block 1
    split_layer = Lambda(lambda x: tf.split(x, num_or_size_splits=3, axis=3))(input_layer)
    conv1 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='valid', activation='relu')(split_layer)
    conv2 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='valid', activation='relu')(split_layer)
    conv3 = Conv2D(filters=32, kernel_size=(5, 5), strides=(1, 1), padding='valid', activation='relu')(split_layer)
    dropout = Dropout(0.5)(Concatenate()([conv1, conv2, conv3]))

    # Block 2
    branch1 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='valid', activation='relu')(dropout)
    branch2 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='valid', activation='relu')(branch1)
    branch2 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='valid', activation='relu')(branch2)
    branch3 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='valid', activation='relu')(dropout)
    branch3 = Conv2D(filters=64, kernel_size=(5, 5), strides=(1, 1), padding='valid', activation='relu')(branch3)
    branch4 = MaxPooling2D(pool_size=(2, 2), strides=2, padding='valid')(dropout)
    branch4 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='valid', activation='relu')(branch4)
    concat_branches = Concatenate()([branch1, branch2, branch3, branch4])

    # Output layer
    flatten_layer = Flatten()(concat_branches)
    dense = Dense(units=10, activation='softmax')(flatten_layer)

    model = keras.Model(inputs=input_layer, outputs=dense)

    return model