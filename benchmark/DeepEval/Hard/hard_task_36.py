from keras.models import Model
from keras.layers import Input, Conv2D, MaxPooling2D, Dropout, Add, GlobalAveragePooling2D, Flatten, Dense

def dl_model():
    
    input_layer = Input(shape=(28,28,1))

    def block(input_tensor):

        conv1 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        conv2 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(conv1)
        conv3 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(conv2)
        pool = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(conv3)
        output_tensor = Dropout(rate=0.5)(pool)

        return output_tensor

    block_output_1 = block(input_tensor=input_layer)
    main_path = block(input_tensor=block_output_1)

    branch_path = Conv2D(filters=main_path.shape[-1], kernel_size=(3, 3), strides=(4, 4), padding='same', activation='relu')(input_layer)
    added = Add()([main_path, branch_path])

    average_pool = GlobalAveragePooling2D()(added)
    flatten = Flatten()(average_pool)
    output_layer = Dense(units=10, activation='softmax')(flatten)
    
    model = Model(inputs=input_layer, outputs=output_layer)

    return model