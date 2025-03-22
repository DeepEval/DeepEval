from keras.models import Model
from keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, Concatenate, Add, Dense


def dl_model():

    input_layer = Input(shape=(32, 32, 3))

    initial_conv = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_layer)

    path1 = Conv2D(filters=64, kernel_size=(3, 3), padding='same', activation='relu')(initial_conv)

    pool2_2 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(initial_conv)  
    conv2_2 = Conv2D(filters=32, kernel_size=(3, 3), padding='same', activation='relu')(pool2_2)
    path2 = UpSampling2D(size=(2, 2))(conv2_2)

    pool2_3 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2),padding='same')(initial_conv)  
    conv2_3 = Conv2D(filters=32, kernel_size=(3, 3), padding='same', activation='relu')(pool2_3)
    path3 = UpSampling2D(size=(2, 2))(conv2_3)

    concat = Concatenate()([path1, path2, path3]) 
    main_path = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(concat)

    branch_path = Conv2D(filters=main_path.shape[-1], kernel_size=(1, 1), padding='same', activation='relu')(initial_conv)
    added = Add()([main_path, branch_path])

    dense_1 = Dense(units=128, activation='relu')(added)
    output_layer = Dense(units=10, activation='softmax')(dense_1)

    model = Model(inputs=input_layer, outputs=output_layer)

    return model
