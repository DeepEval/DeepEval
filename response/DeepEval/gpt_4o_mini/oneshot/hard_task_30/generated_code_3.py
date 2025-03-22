import keras
from keras.layers import Input, Conv2D, Add, Lambda, DepthwiseConv2D, Concatenate, Flatten, Dense
import tensorflow as tf

def dl_model():     
    input_layer = Input(shape=(32, 32, 3))

    # First Block: Dual-path structure
    # Main path
    main_path = Conv2D(filters=32, kernel_size=(3, 3), padding='same', activation='relu')(input_layer)
    main_path = Conv2D(filters=64, kernel_size=(3, 3), padding='same', activation='relu')(main_path)
    main_path = Conv2D(filters=3, kernel_size=(3, 3), padding='same', activation='relu')(main_path)
    
    # Branch path
    branch_path = input_layer

    # Combine both paths
    combined_path = Add()([main_path, branch_path])

    # Second Block: Splitting the input into three groups
    split = Lambda(lambda x: tf.split(x, num_or_size_splits=3, axis=-1))(combined_path)

    # Depthwise separable convolution for each group with different kernel sizes
    path1 = DepthwiseConv2D(kernel_size=(1, 1), padding='same', activation='relu')(split[0])
    path2 = DepthwiseConv2D(kernel_size=(3, 3), padding='same', activation='relu')(split[1])
    path3 = DepthwiseConv2D(kernel_size=(5, 5), padding='same', activation='relu')(split[2])

    # Concatenate outputs from the three paths
    concatenated = Concatenate()([path1, path2, path3])

    # Flatten and fully connected layers
    flatten_layer = Flatten()(concatenated)
    dense1 = Dense(units=128, activation='relu')(flatten_layer)
    output_layer = Dense(units=10, activation='softmax')(dense1)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model