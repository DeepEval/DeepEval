import keras
from keras.layers import Input, Lambda, Conv2D, MaxPooling2D, Concatenate, BatchNormalization, Flatten, Dense

def dl_model():

    input_layer = Input(shape=(32, 32, 3))
    # Split the input image into three channel groups
    channel_groups = Lambda(lambda x: tf.split(x, num_or_size_splits=3, axis=3))(input_layer)

    # Feature extraction for each group
    group1 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(channel_groups[0])
    group2 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(channel_groups[1])
    group3 = Conv2D(filters=32, kernel_size=(5, 5), strides=(1, 1), padding='same', activation='relu')(channel_groups[2])

    # Concatenate the outputs from the three groups
    concat = Concatenate()([group1, group2, group3])

    # Batch normalization and flattening
    batch_norm = BatchNormalization()(concat)
    flatten_layer = Flatten()(batch_norm)

    # Fully connected layers
    dense1 = Dense(units=128, activation='relu')(flatten_layer)
    dense2 = Dense(units=64, activation='relu')(dense1)
    output_layer = Dense(units=10, activation='softmax')(dense2)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model