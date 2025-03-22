import keras
from keras.layers import Input, Conv2D, DepthwiseConv2D, Concatenate, Lambda, Reshape, Permute

def dl_model():
    
    input_layer = Input(shape=(28,28,1))

    # Block 1
    main_path = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_layer)
    main_path = DepthwiseConv2D(kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(main_path)
    main_path = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(main_path)

    branch_path = DepthwiseConv2D(kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_layer)
    branch_path = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(branch_path)

    merged_path = Concatenate(axis=-1)([main_path, branch_path])

    # Block 2
    shape = Lambda(lambda x: tf.shape(x)) (merged_path)
    reshaped = Reshape(target_shape=(shape[1], shape[2], 4, 16))(merged_path) 
    permuted = Permute(axes=[0, 1, 3, 2])(reshaped)
    reshaped_back = Reshape(target_shape=(shape[1], shape[2], 64))(permuted)

    flatten = Flatten()(reshaped_back)
    output_layer = Dense(units=10, activation='softmax')(flatten)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model