from tensorflow.keras.layers import Input, Lambda, Conv2D, BatchNormalization, Activation, Dropout, concatenate, Dense
from tensorflow.keras.models import Model
from tensorflow.keras.datasets import cifar10
from tensorflow.keras import backend as K

def split_rgb_channels(x):
    x_r, x_g, x_b = K.split(x, num_or_size_splits=3, axis=3)
    return x_r, x_g, x_b

def merge_rgb_channels(x):
    return K.concatenate(x, axis=3)

def residual_network(filters, kernel_size, strides, activation):
    def layer(input_tensor):
        x = Conv2D(filters, kernel_size, strides=strides, padding='same')(input_tensor)
        x = BatchNormalization()(x)
        x = Activation(activation)(x)
        x = Conv2D(filters, kernel_size, strides=1, padding='same')(x)
        x = BatchNormalization()(x)
        if strides != 1 or K.int_shape(input_tensor)[3] != filters:
            shortcut = Conv2D(filters, kernel_size, strides=strides, padding='same')(input_tensor)
            shortcut = BatchNormalization()(shortcut)
        else:
            shortcut = input_tensor
        x = layers.add([x, shortcut])
        x = Activation(activation)(x)
        return x
    return layer

def residual_block(x, filters, kernel_size, strides, dropout):
    pathway = residual_network(filters, kernel_size, strides, 'relu')(x)
    pathway = Dropout(dropout)(pathway)
    shortcut = residual_network(filters, kernel_size, strides, 'relu')(pathway)
    shortcut = Dropout(dropout)(shortcut)
    return shortcut

def dl_model():
    input_image = Input(shape=(32, 32, 3))

    x_r, x_g, x_b = Lambda(split_rgb_channels)(input_image)

    # Main Pathway
    x_r = residual_block(x_r, 64, (1, 1), (1, 1), 0.2)
    x_r = residual_block(x_r, 64, (3, 3), (1, 1), 0.2)

    x_g = residual_block(x_g, 64, (1, 1), (1, 1), 0.2)
    x_g = residual_block(x_g, 64, (3, 3), (1, 1), 0.2)

    x_b = residual_block(x_b, 64, (1, 1), (1, 1), 0.2)
    x_b = residual_block(x_b, 64, (3, 3), (1, 1), 0.2)

    main_pathway = Lambda(merge_rgb_channels)([x_r, x_g, x_b])

    # Branch Pathway
    branch_pathway = Conv2D(64, (1, 1), strides=(1, 1), padding='same')(input_image)

    branch_pathway = residual_block(branch_pathway, 64, (3, 3), (1, 1), 0.2)

    # Concatenation and Addition
    main_pathway = concatenate([main_pathway, branch_pathway])

    # Classification Layer
    main_pathway = Dense(10, activation='softmax')(main_pathway)

    model = Model(inputs=input_image, outputs=main_pathway)

    return model

if __name__ == '__main__':
    model = dl_model()
    model.summary()