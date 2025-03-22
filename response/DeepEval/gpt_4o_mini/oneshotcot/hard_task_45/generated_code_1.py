import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, Concatenate, Lambda, DepthwiseConv2D
import tensorflow as tf

def dl_model():
    input_layer = Input(shape=(32, 32, 3))  # CIFAR-10 images are 32x32 with 3 color channels

    # First block
    split_tensor = Lambda(lambda x: tf.split(x, num_or_size_splits=3, axis=-1))(input_layer)
    
    path1 = DepthwiseConv2D(kernel_size=(1, 1), padding='same', activation='relu')(split_tensor[0])
    path2 = DepthwiseConv2D(kernel_size=(3, 3), padding='same', activation='relu')(split_tensor[1])
    path3 = DepthwiseConv2D(kernel_size=(5, 5), padding='same', activation='relu')(split_tensor[2])
    
    block1_output = Concatenate()([path1, path2, path3])

    # Second block
    branch1 = Conv2D(filters=32, kernel_size=(1, 1), padding='same', activation='relu')(block1_output)
    branch2 = Conv2D(filters=32, kernel_size=(1, 1), padding='same', activation='relu')(Conv2D(filters=32, kernel_size=(3, 3), padding='same', activation='relu')(block1_output))
    branch3 = Conv2D(filters=32, kernel_size=(3, 3), padding='same', activation='relu')(Conv2D(filters=32, kernel_size=(3, 3), padding='same', activation='relu')(block1_output))
    branch4 = Conv2D(filters=32, kernel_size=(1, 1), padding='same', activation='relu')(block1_output)
    branch5 = MaxPooling2D(pool_size=(2, 2), padding='same')(block1_output)
    branch5 = Conv2D(filters=32, kernel_size=(1, 1), padding='same', activation='relu')(branch5)

    block2_output = Concatenate()([branch1, branch2, branch3, branch4, branch5])

    # Flattening and Dense layers
    flatten_layer = Flatten()(block2_output)
    dense_layer = Dense(units=128, activation='relu')(flatten_layer)
    output_layer = Dense(units=10, activation='softmax')(dense_layer)  # 10 classes for CIFAR-10

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model