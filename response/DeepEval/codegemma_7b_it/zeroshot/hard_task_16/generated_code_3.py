from tensorflow.keras import backend as K
from tensorflow.keras.layers import Conv2D, Lambda, Input, MaxPooling2D, Dense, Reshape, Concatenate, Add
from tensorflow.keras.models import Model
from tensorflow.keras.utils import plot_model

def block_1(x):
    x = Conv2D(32, (1, 1), padding='same')(x)
    x = Conv2D(64, (3, 3), padding='same')(x)
    x = Conv2D(32, (1, 1), padding='same')(x)
    return x

def transition_conv(x):
    x = Conv2D(128, (1, 1), padding='same')(x)
    return x

def block_2(x):
    branch_drop = K.Dropout(0.5)(x)
    branch_pool = MaxPooling2D()(branch_drop)
    branch_pool = Conv2D(128, (1, 1), padding='same')(branch_pool)
    branch_pool = K.Dropout(0.5)(branch_pool)
    branch_pool = Conv2D(128, (1, 1), padding='same')(branch_pool)

    branch_dense_1 = Dense(128)(x)
    branch_dense_1 = Reshape((1, 1, 128))(branch_dense_1)
    branch_dense_1 = K.repeat_elem(branch_dense_1, K.shape(x)[1] * K.shape(x)[2])
    branch_dense_1 = Reshape((K.shape(x)[1], K.shape(x)[2], 128))(branch_dense_1)

    branch_dense_2 = Dense(128)(x)
    branch_dense_2 = Reshape((1, 1, 128))(branch_dense_2)
    branch_dense_2 = K.repeat_elem(branch_dense_2, K.shape(x)[1] * K.shape(x)[2])
    branch_dense_2 = Reshape((K.shape(x)[1], K.shape(x)[2], 128))(branch_dense_2)
    branch_dense_2 = K.permute_dimensions(branch_dense_2, (0, 3, 1, 2))
    branch_dense_2 = Conv2D(128, (1, x.shape[1]), padding='valid')(branch_dense_2)
    branch_dense_2 = Reshape((K.shape(x)[1], K.shape(x)[2], 128))(branch_dense_2)

    branch_concat = Concatenate(axis=3)([branch_dense_1, branch_dense_2, branch_pool])
    return branch_concat

def dl_model():
    img_input = Input(shape=(32, 32, 3))

    x = block_1(img_input)
    x = transition_conv(x)
    x = block_2(x)

    branch_direct = Conv2D(10, (1, 1), padding='same')(img_input)
    branch_direct = Flatten()(branch_direct)

    branch_main = Flatten()(x)
    branch_main = Dense(10, activation='softmax')(branch_main)

    output = Add()([branch_main, branch_direct])

    model = Model(inputs=[img_input], outputs=[output])
    return model

model = dl_model()
model.summary()