import keras
from keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, Add, Flatten, Dense

def dl_model():
    input_layer = Input(shape=(32, 32, 3))

    # Main Path
    x = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), activation='relu')(input_layer)
    branch1 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), activation='relu')(x)
    branch2 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(x)
    branch2 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), activation='relu')(branch2)
    branch2 = UpSampling2D(size=(2, 2))(branch2)
    branch3 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(x)
    branch3 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), activation='relu')(branch3)
    branch3 = UpSampling2D(size=(2, 2))(branch3)
    main_path = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), activation='relu')(Add()([branch1, branch2, branch3]))

    # Branch Path
    branch_path = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), activation='relu')(input_layer)

    # Concatenate and classify
    merged = Add()([main_path, branch_path])
    flatten_layer = Flatten()(merged)
    dense1 = Dense(units=128, activation='relu')(flatten_layer)
    output_layer = Dense(units=10, activation='softmax')(dense1)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model