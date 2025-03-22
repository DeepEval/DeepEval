import keras
from keras.layers import Input, Lambda, Conv2D, MaxPooling2D, Concatenate, BatchNormalization, Flatten, Dense

def dl_model():

    input_layer = Input(shape=(32, 32, 3))
    
    # Split the input image into three channel groups
    x = Lambda(lambda x: tf.split(x, num_or_size_splits=3, axis=3))(input_layer)

    # Feature extraction for each group
    conv1_1x1 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(x[0])
    conv3_3x3 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(x[1])
    conv5_5x5 = Conv2D(filters=64, kernel_size=(5, 5), strides=(1, 1), padding='same', activation='relu')(x[2])
    maxpool = MaxPooling2D(pool_size=(2, 2), strides=2, padding='valid')(x[0])

    # Concatenate the outputs
    concat = Concatenate()([conv1_1x1, conv3_3x3, conv5_5x5, maxpool])

    # Batch normalization
    bath_norm = BatchNormalization()(concat)

    # Flatten and fully connected layers
    flatten_layer = Flatten()(bath_norm)
    dense1 = Dense(units=128, activation='relu')(flatten_layer)
    dense2 = Dense(units=64, activation='relu')(dense1)
    output_layer = Dense(units=10, activation='softmax')(dense2)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model