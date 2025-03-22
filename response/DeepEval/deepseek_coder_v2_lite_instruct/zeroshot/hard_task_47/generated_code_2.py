import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Lambda, Conv2D, BatchNormalization, ReLU, Add, Concatenate, GlobalAveragePooling2D, Dense

def dl_model():
    input_layer = Input(shape=(32, 32, 3))

    # First Block
    def split_and_apply(x, kernel_size):
        split_layers = tf.split(x, num_or_size_splits=3, axis=-1)
        outputs = []
        for split_layer in split_layers:
            if kernel_size == 1:
                conv = Conv2D(filters=split_layer.shape[-1], kernel_size=(1, 1), padding='same')(split_layer)
            elif kernel_size == 3:
                conv = Conv2D(filters=split_layer.shape[-1], kernel_size=(3, 3), padding='same')(split_layer)
            elif kernel_size == 5:
                conv = Conv2D(filters=split_layer.shape[-1], kernel_size=(5, 5), padding='same')(split_layer)
            conv = BatchNormalization()(conv)
            conv = ReLU()(conv)
            outputs.append(conv)
        return Concatenate(axis=-1)(outputs)

    first_block = Lambda(lambda x: split_and_apply(x, 1), output_shape=lambda shape: (shape[0], shape[1], shape[2] // 3 * 3))(input_layer)
    first_block = Lambda(lambda x: split_and_apply(x, 3), output_shape=lambda shape: (shape[0], shape[1], shape[2] // 3 * 3))(first_block)
    first_block = Lambda(lambda x: split_and_apply(x, 5), output_shape=lambda shape: (shape[0], shape[1], shape[2] // 3 * 3))(first_block)

    # Second Block
    def branch_block(x, kernel_size1, kernel_size2=None, kernel_size3=None):
        if kernel_size2 is not None and kernel_size3 is not None:
            x = Conv2D(filters=x.shape[-1], kernel_size=(1, 1), padding='same')(x)
            x = Conv2D(filters=x.shape[-1], kernel_size=(kernel_size2, kernel_size2), padding='same')(x)
            x = Conv2D(filters=x.shape[-1], kernel_size=(kernel_size3, kernel_size3), padding='same')(x)
            x = Conv2D(filters=x.shape[-1], kernel_size=(3, 3), padding='same')(x)
        elif kernel_size2 is not None:
            x = Conv2D(filters=x.shape[-1], kernel_size=(1, 1), padding='same')(x)
            x = Conv2D(filters=x.shape[-1], kernel_size=(kernel_size2, kernel_size2), padding='same')(x)
            x = Conv2D(filters=x.shape[-1], kernel_size=(3, 3), padding='same')(x)
        else:
            x = Conv2D(filters=x.shape[-1], kernel_size=(kernel_size1, kernel_size1), padding='same')(x)
        x = BatchNormalization()(x)
        x = ReLU()(x)
        return x

    branch1 = branch_block(first_block, 1)
    branch2 = branch_block(first_block, 3)
    branch3 = branch_block(first_block, 1, 7, 7)
    branch4 = branch_block(first_block, 3, None, None)
    branch5 = GlobalAveragePooling2D()(first_block)
    branch5 = Dense(128, activation='relu')(branch5)

    second_block = Concatenate(axis=-1)([branch1, branch2, branch3, branch4, branch5])

    # Output Layer
    output_layer = Dense(10, activation='softmax')(second_block)

    model = Model(inputs=input_layer, outputs=output_layer)
    return model

# Example usage:
# model = dl_model()
# model.summary()