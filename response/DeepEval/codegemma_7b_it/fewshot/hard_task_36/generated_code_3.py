import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Lambda, Flatten, Dense, Dropout, Concatenate

def dl_model():

    input_layer = Input(shape=(28, 28, 1))

    def specialized_block(input_tensor):
        conv1 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        conv2 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(conv1)
        max_pooling = MaxPooling2D(pool_size=(2, 2), strides=(1, 1), padding='same')(conv2)
        dropout = Dropout(rate=0.5)(max_pooling)
        return dropout

    main_path = specialized_block(input_tensor=input_layer)
    branch_path = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_layer)
    fused = Concatenate()([main_path, branch_path])
    avg_pooling = GlobalAveragePooling2D()(fused)
    flatten = Flatten()(avg_pooling)
    output_layer = Dense(units=10, activation='softmax')(flatten)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model