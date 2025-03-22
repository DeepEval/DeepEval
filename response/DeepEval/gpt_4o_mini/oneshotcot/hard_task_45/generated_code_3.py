import keras
from keras.layers import Input, DepthwiseConv2D, Lambda, Concatenate, Flatten, Dense
import tensorflow as tf

def dl_model():
    input_layer = Input(shape=(32, 32, 3))

    # First Block
    split_tensor = Lambda(lambda x: tf.split(x, num_or_size_splits=3, axis=-1))(input_layer)

    path1 = DepthwiseConv2D(kernel_size=(1, 1), padding='same', activation='relu')(split_tensor[0])
    path2 = DepthwiseConv2D(kernel_size=(3, 3), padding='same', activation='relu')(split_tensor[1])
    path3 = DepthwiseConv2D(kernel_size=(5, 5), padding='same', activation='relu')(split_tensor[2])
    
    block1_output = Concatenate()([path1, path2, path3])

    # Second Block
    path1 = DepthwiseConv2D(kernel_size=(1, 1), padding='same', activation='relu')(block1_output)
    
    path2 = DepthwiseConv2D(kernel_size=(1, 1), padding='same', activation='relu')(block1_output)
    path2 = DepthwiseConv2D(kernel_size=(3, 3), padding='same', activation='relu')(path2)

    path3 = DepthwiseConv2D(kernel_size=(3, 3), padding='same', activation='relu')(block1_output)
    path3 = DepthwiseConv2D(kernel_size=(3, 3), padding='same', activation='relu')(path3)

    path4 = DepthwiseConv2D(kernel_size=(1, 1), padding='same', activation='relu')(block1_output)

    path5 = MaxPooling2D(pool_size=(2, 2), strides=(1, 1), padding='same')(block1_output)
    path5 = DepthwiseConv2D(kernel_size=(1, 1), padding='same', activation='relu')(path5)

    block2_output = Concatenate()([path1, path2, path3, path4, path5])

    # Final layers
    flatten_layer = Flatten()(block2_output)
    dense_layer = Dense(units=128, activation='relu')(flatten_layer)
    output_layer = Dense(units=10, activation='softmax')(dense_layer)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model