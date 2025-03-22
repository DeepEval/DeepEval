import keras
from keras.layers import Input, Lambda, Conv2D, MaxPooling2D, Dropout, Concatenate, Dense, Add, Flatten

def dl_model():
    input_layer = Input(shape=(32, 32, 3))

    # Split the input into three groups
    inputs_groups = Lambda(lambda x: tf.split(value=x, num_or_size_splits=3, axis=-1))(input_layer)

    # Main pathway
    conv1_main = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(inputs_groups[0])
    conv2_main = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(conv1_main)
    dropout_main = Dropout(rate=0.2)(conv2_main)

    conv3_main = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(inputs_groups[1])
    conv4_main = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(conv3_main)
    dropout_main = Dropout(rate=0.2)(conv4_main)

    conv5_main = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(inputs_groups[2])
    conv6_main = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(conv5_main)
    dropout_main = Dropout(rate=0.2)(conv6_main)

    concat_main = Concatenate()([dropout_main, dropout_main, dropout_main])

    # Branch pathway
    conv1_branch = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_layer)

    # Flatten and reshape
    flatten_branch = Flatten()(conv1_branch)
    reshape_branch = Reshape(target_shape=(1, 1, 64))(flatten_branch)

    # Add the outputs from the main and branch pathways
    output = Add()([concat_main, reshape_branch])

    # Fully connected layer
    dense = Dense(units=10, activation='softmax')(output)

    model = keras.Model(inputs=input_layer, outputs=dense)

    return model