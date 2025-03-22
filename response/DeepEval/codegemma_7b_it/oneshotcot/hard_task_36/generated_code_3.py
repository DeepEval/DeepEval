import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, BatchNormalization, Flatten, Dense, Dropout

def dl_model():     

    input_layer = Input(shape=(28, 28, 1))
    conv = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='valid', activation='sigmoid')(input_layer)
    max_pooling = MaxPooling2D(pool_size=(2, 2), strides=2, padding='valid')(conv)

    def specialized_block(input_tensor):

        conv1 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        conv2 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(conv1)
        max_pooling = MaxPooling2D(pool_size=(2, 2), strides=2, padding='same')(conv2)
        dropout = Dropout(rate=0.5)(max_pooling)

        return dropout

    def branch_pathway(input_tensor):

        conv = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_tensor)

        return conv

    main_pathway_output = specialized_block(input_tensor=max_pooling)
    branch_pathway_output = branch_pathway(input_tensor=max_pooling)

    combined_output = keras.layers.Add()([main_pathway_output, branch_pathway_output])

    gap = keras.layers.GlobalAveragePooling2D()(combined_output)
    flatten_layer = Flatten()(gap)
    output_layer = Dense(units=10, activation='softmax')(flatten_layer)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model