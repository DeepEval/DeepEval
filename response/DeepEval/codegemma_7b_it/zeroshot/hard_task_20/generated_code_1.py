from keras.layers import Input, Lambda, Conv2D, concatenate, Dropout, Flatten, Dense
from keras.models import Model
from keras.utils import to_categorical
from keras.datasets import cifar10
from keras.regularizers import l2
import tensorflow as tf

def main_path(input_tensor):
    tower_1 = Conv2D(64, (1, 1), padding='same', activation='relu')(input_tensor)
    tower_2 = Conv2D(64, (3, 3), padding='same', activation='relu')(input_tensor)
    tower_3 = Conv2D(64, (5, 5), padding='same', activation='relu')(input_tensor)
    return concatenate([tower_1, tower_2, tower_3], axis=3)

def branch_path(input_tensor):
    tower = Conv2D(64, (1, 1), padding='same', activation='relu')(input_tensor)
    return tower

def identity_block(input_tensor, kernel_size, filters):
    x = Conv2D(filters, (kernel_size, kernel_size), padding='same', activation='relu')(input_tensor)
    x = Dropout(0.1)(x)
    x = Conv2D(filters, (kernel_size, kernel_size), padding='same', activation='relu')(x)
    x = Dropout(0.1)(x)
    x = concatenate([x, input_tensor], axis=3)
    return x

def projection_block(input_tensor, kernel_size, filters, s=2):
    x = Conv2D(filters, (kernel_size, kernel_size), strides=(s, s), padding='valid', activation='relu')(input_tensor)
    x = Dropout(0.1)(x)
    x = Conv2D(filters, (kernel_size, kernel_size), padding='same', activation='relu')(x)
    x = Dropout(0.1)(x)
    shortcut = Conv2D(filters, (1, 1), strides=(s, s), padding='valid')(input_tensor)
    x = concatenate([x, shortcut], axis=3)
    return x

def residual_group(input_tensor, kernel_size, filters, reps, s=1):
    x = input_tensor
    if s > 1:
        x = projection_block(input_tensor, kernel_size, filters, s)
    else:
        x = identity_block(input_tensor, kernel_size, filters)
    for i in range(reps - 1):
        x = identity_block(x, kernel_size, filters)
    return x

def dl_model():
    input_tensor = Input(shape=(32, 32, 3))
    x = input_tensor
    x = Lambda(lambda y: tf.split(y, 3, axis=3))(x)

    main_output = main_path(x)
    branch_output = branch_path(input_tensor)

    fused_features = Lambda(lambda y: tf.add(y))(main_output + branch_output)

    x = residual_group(fused_features, 1, 64, 3)
    x = residual_group(x, 3, 128, 4)
    x = residual_group(x, 5, 256, 6)
    x = residual_group(x, 7, 512, 3)

    x = Flatten()(x)
    x = Dense(256, activation='relu')(x)
    x = Dropout(0.5)(x)
    output = Dense(10, activation='softmax')(x)

    model = Model(inputs=input_tensor, outputs=output)

    return model