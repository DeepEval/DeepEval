import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

def basic_block(filters, stage, block):
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    def shortcut(input_tensor):
        x = layers.Conv2D(filters, (1, 1), strides=(1, 1), kernel_initializer='he_normal', name=conv_name_base + '2')(input_tensor)
        x = layers.BatchNormalization(name=bn_name_base + '2')(x)

        return x

    def main_path(input_tensor):
        x = layers.Conv2D(filters, (3, 3), strides=(1, 1), padding='same', kernel_initializer='he_normal', name=conv_name_base + '1')(input_tensor)
        x = layers.BatchNormalization(name=bn_name_base + '1')(x)
        x = layers.Activation('relu')(x)

        x = layers.Conv2D(filters, (3, 3), strides=(1, 1), padding='same', kernel_initializer='he_normal', name=conv_name_base + '2')(x)
        x = layers.BatchNormalization(name=bn_name_base + '2')(x)

        return x

    def branch(input_tensor):
        x = layers.Conv2D(filters, (1, 1), strides=(1, 1), kernel_initializer='he_normal', name=conv_name_base + '2')(input_tensor)
        x = layers.BatchNormalization(name=bn_name_base + '2')(x)

        return x

    input_tensor = layers.Input(shape=(32, 32, 3))
    x = layers.Conv2D(16, (1, 1), strides=(1, 1), padding='same', kernel_initializer='he_normal')(input_tensor)

    shortcut_output = shortcut(x)

    x = main_path(x)

    x = layers.Add()([x, shortcut_output])
    x = layers.Activation('relu')(x)

    branch_output = branch(input_tensor)

    x = layers.Add()([x, branch_output])
    x = layers.Activation('relu')(x)

    return keras.Model(inputs=input_tensor, outputs=x)

def dl_model():
    # Load CIFAR-10 dataset
    (x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()

    # Preprocess data
    x_train = x_train.astype('float32') / 255.0
    x_test = x_test.astype('float32') / 255.0

    # Define model architecture
    inputs = keras.Input(shape=(32, 32, 3))
    x = layers.Conv2D(16, (1, 1), strides=(1, 1), padding='same', kernel_initializer='he_normal')(inputs)

    # First level
    basic_block_1 = basic_block(16, 1, 'a')
    x = basic_block_1(x)

    # Second level
    basic_block_2 = basic_block(32, 2, 'a')
    x = basic_block_2(x)

    # Third level
    x = layers.Conv2D(64, (1, 1), strides=(1, 1), padding='same', kernel_initializer='he_normal')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)

    x = layers.AveragePooling2D()(x)

    x = layers.Flatten()(x)
    outputs = layers.Dense(10, activation='softmax')(x)

    model = keras.Model(inputs=inputs, outputs=outputs)

    return model