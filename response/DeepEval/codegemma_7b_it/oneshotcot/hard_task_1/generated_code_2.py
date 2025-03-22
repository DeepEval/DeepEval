import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, BatchNormalization, Flatten, Dense, GlobalAveragePooling2D, GlobalMaxPooling2D, Multiply

def dl_model():

    input_layer = Input(shape=(32, 32, 3))
    conv1 = Conv2D(filters=16, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_layer)

    # Block 1: Channel Attention
    path1 = GlobalAveragePooling2D()(conv1)
    path1 = Dense(units=16, activation='relu')(path1)
    path1 = Dense(units=16, activation='sigmoid')(path1)
    path1 = Reshape((1, 1, 16))(path1)
    path1 = Multiply()([conv1, path1])

    path2 = GlobalMaxPooling2D()(conv1)
    path2 = Dense(units=16, activation='relu')(path2)
    path2 = Dense(units=16, activation='sigmoid')(path2)
    path2 = Reshape((1, 1, 16))(path2)
    path2 = Multiply()([conv1, path2])

    conv_concat = Concatenate()([path1, path2])
    conv_concat = Conv2D(filters=16, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(conv_concat)

    # Block 2: Spatial Attention
    path3 = AveragePooling2D()(conv1)
    path4 = MaxPooling2D()(conv1)
    concat = Concatenate()([path3, path4])
    concat = Conv2D(filters=16, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(concat)
    concat = Conv2D(filters=1, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='sigmoid')(concat)
    concat = Flatten()(concat)
    concat = RepeatVector(x=conv1.shape[1] * conv1.shape[2])(concat)
    concat = Reshape((conv1.shape[1], conv1.shape[2], 1))(concat)
    concat = Multiply()([conv1, concat])

    # Final Path
    final_path = Concatenate()([conv_concat, concat])
    final_path = Conv2D(filters=16, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(final_path)

    # Classification
    flatten = Flatten()(final_path)
    dense = Dense(units=10, activation='softmax')(flatten)

    model = keras.Model(inputs=input_layer, outputs=dense)

    return model