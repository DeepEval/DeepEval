import keras
from keras.layers import Lambda, Input, Conv2D, MaxPooling2D, Concatenate, BatchNormalization, Flatten, Dense

def dl_model():
    def block1(input_tensor):
        # Split the input into three groups
        split1 = Lambda(lambda x: tf.split(x, [16, 16, 16], axis=2))(input_tensor)
        split1 = [Conv2D(filters=32, kernel_size=(1, 1), padding='same', activation='relu', name='sepconv1_%d' % i)(split1[i]) for i in range(3)]
        batch_norm1 = [BatchNormalization(name='batchnorm1_%d' % i)(split1[i]) for i in range(3)]
        concat = Concatenate(name='concat1')(batch_norm1)
        return concat

    def block2(concat):
        # Path 1: 1x1 convolution
        path1 = Conv2D(filters=64, kernel_size=(1, 1), padding='same', activation='relu', name='path1')(concat)
        # Path 2: 3x3 average pooling, 1x1 convolution
        path2 = MaxPooling2D(pool_size=(3, 3), strides=(1, 1), padding='same')(concat)
        path2 = Conv2D(filters=64, kernel_size=(1, 1), padding='same', activation='relu', name='path2')(path2)
        # Path 3: 1x1 convolution, split into two paths
        path3 = Conv2D(filters=64, kernel_size=(1, 1), padding='same', activation='relu', name='path3')(concat)
        path3_subpath1 = Conv2D(filters=64, kernel_size=(1, 3), padding='same', activation='relu', name='path3_subpath1')(path3)
        path3_subpath2 = Conv2D(filters=64, kernel_size=(3, 1), padding='same', activation='relu', name='path3_subpath2')(path3)
        # Path 4: 3x3 convolution, split into two paths
        path4 = Conv2D(filters=64, kernel_size=(3, 1), padding='same', activation='relu', name='path4')(concat)
        path4_subpath1 = Conv2D(filters=64, kernel_size=(1, 3), padding='same', activation='relu', name='path4_subpath1')(path4)
        path4_subpath2 = Conv2D(filters=64, kernel_size=(3, 1), padding='same', activation='relu', name='path4_subpath2')(path4)
        concat2 = Concatenate(name='concat2')(
            [path1, path2, path3, path4, path3_subpath1, path3_subpath2, path4_subpath1, path4_subpath2])
        return concat2

    input_layer = Input(shape=(32, 32, 3))
    block1_output = block1(input_layer)
    block2_output = block2(block1_output)
    flatten = Flatten()(block2_output)
    dense = Dense(units=128, activation='relu')(flatten)
    output_layer = Dense(units=10, activation='softmax')(dense)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model