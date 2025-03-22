from keras.models import Model
from keras.layers import Input, Conv2D, MaxPooling2D, GlobalAveragePooling2D, Dense, Reshape, Multiply, Add


def dl_model():

    input_layer = Input(shape=(32, 32, 3))
    
    input_channels = input_layer.shape[-1]
    
    conv1 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_layer)
    conv2 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(conv1)
    conv3 = Conv2D(filters=input_channels, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(conv2)
    main_path = MaxPooling2D(pool_size=(2, 2), strides=(1, 1), padding='same')(conv3)

    squeeze = GlobalAveragePooling2D()(input_layer)
    excitation = Dense(units=input_channels // 3, activation='relu')(squeeze)
    excitation = Dense(units=input_channels, activation='sigmoid')(excitation)
    excitation = Reshape(target_shape=(1, 1, input_channels))(excitation)
    branch_path = Multiply()([input_layer, excitation])

    added = Add()([main_path, branch_path])

    dense_2 = Dense(units=64, activation='relu')(added)
    output_layer = Dense(units=10, activation='softmax')(dense_2)

    model = Model(inputs=input_layer, outputs=output_layer)

    return model
