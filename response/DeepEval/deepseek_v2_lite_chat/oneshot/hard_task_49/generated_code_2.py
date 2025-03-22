import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, Lambda, BatchNormalization, Flatten, Dense

def dl_model():

    input_layer = Input(shape=(28, 28, 1))
    pool1 = MaxPooling2D(pool_size=(1, 1), strides=1, padding='valid')(input_layer)
    pool2 = MaxPooling2D(pool_size=(2, 2), strides=2, padding='valid')(input_layer)
    pool3 = MaxPooling2D(pool_size=(4, 4), strides=4, padding='valid')(input_layer)

    flatten_pooled_outputs = Flatten()(Concatenate()([pool1, pool2, pool3]))

    dense_layer = Dense(units=128, activation='relu')(flatten_pooled_outputs)
    reshape_layer = Lambda(lambda x: keras.backend.reshape(x, (-1, 16)))(dense_layer)

    def second_block(input_tensor):
        split_layer = Lambda(lambda x: keras.backend.split(x, 4, axis=-1))(input_tensor)

        path1 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(split_layer[0])
        path2 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(split_layer[1])
        path3 = Conv2D(filters=64, kernel_size=(5, 5), strides=(1, 1), padding='same', activation='relu')(split_layer[2])
        path4 = Conv2D(filters=64, kernel_size=(7, 7), strides=(1, 1), padding='same', activation='relu')(split_layer[3])

        concatenated_outputs = Concatenate()(path1, path2, path3, path4)
        return concatenated_outputs

    second_block_output = second_block(reshape_layer)
    batch_norm = BatchNormalization()(second_block_output)
    dense_output = Dense(units=10, activation='softmax')(batch_norm)

    model = keras.Model(inputs=input_layer, outputs=dense_output)

    return model

model = dl_model()
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])