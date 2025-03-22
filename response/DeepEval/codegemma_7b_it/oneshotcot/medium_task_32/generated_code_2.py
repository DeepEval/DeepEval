import keras
from keras.layers import Input, Lambda, Conv2D, MaxPooling2D, Concatenate, BatchNormalization, Flatten, Dense

def dl_model():
    input_layer = Input(shape=(32, 32, 3))

    # Split the input into three groups
    x = Lambda(lambda x: tf.split(x, num_or_size_splits=3, axis=-1))(input_layer)

    # Feature extraction for each group
    group1 = Conv2D(filters=32, kernel_size=(1, 1), padding='same', activation='relu')(x[0])
    group2 = Conv2D(filters=32, kernel_size=(3, 3), padding='same', activation='relu')(x[1])
    group3 = Conv2D(filters=32, kernel_size=(5, 5), padding='same', activation='relu')(x[2])

    # Concatenate the outputs and fuse them
    fuse = Concatenate()([group1, group2, group3])

    # Flatten and pass through fully connected layers
    flatten = Flatten()(fuse)
    dense1 = Dense(units=64, activation='relu')(flatten)
    dense2 = Dense(units=32, activation='relu')(dense1)
    output_layer = Dense(units=10, activation='softmax')(dense2)

    # Create the model
    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model