import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, BatchNormalization, Flatten, Dense, Lambda

def dl_model():     

    input_layer = Input(shape=(32, 32, 3))
    
    # Block 1
    split_input = Lambda(lambda x: tf.split(x, num_or_size_splits=3, axis=3))(input_layer)
    conv1 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(split_input[0])
    conv2 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(split_input[1])
    conv3 = Conv2D(filters=32, kernel_size=(5, 5), strides=(1, 1), padding='same', activation='relu')(split_input[2])
    concat_block1 = Concatenate()([conv1, conv2, conv3])
    dropout_block1 = Dropout(0.2)(concat_block1)

    # Block 2
    branch1 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(dropout_block1)
    branch2 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(dropout_block1)
    branch3 = Conv2D(filters=32, kernel_size=(5, 5), strides=(1, 1), padding='same', activation='relu')(dropout_block1)
    branch4 = MaxPooling2D(pool_size=(2, 2), strides=2, padding='valid')(dropout_block1)
    branch4 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(branch4)
    concat_block2 = Concatenate()([branch1, branch2, branch3, branch4])

    # Output layer
    flatten_layer = Flatten()(concat_block2)
    dense_layer = Dense(units=10, activation='softmax')(flatten_layer)

    model = keras.Model(inputs=input_layer, outputs=dense_layer)

    return model