import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, Dropout, BatchNormalization, Flatten, Dense, Lambda

def dl_model():

    input_layer = Input(shape=(32, 32, 3))
    split_inputs = Lambda(lambda x: tf.split(x, num_or_size_splits=3, axis=3))(input_layer)

    def grouped_convolutions(input_tensor):

        path1 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        path1 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(path1)
        path1 = Dropout(0.2)(path1)

        path2 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        path2 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(path2)
        path2 = Dropout(0.2)(path2)

        path3 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        path3 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(path3)
        path3 = Dropout(0.2)(path3)

        return Concatenate()([path1, path2, path3])

    main_pathway = grouped_convolutions(input_tensor=split_inputs)

    branch_pathway = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(split_inputs)
    branch_pathway = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(branch_pathway)

    combined_output = Add()([main_pathway, branch_pathway])
    combined_output = BatchNormalization()(combined_output)
    combined_output = Flatten()(combined_output)
    output_layer = Dense(units=10, activation='softmax')(combined_output)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model