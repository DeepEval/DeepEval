import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Add, Flatten, Dense, Lambda, DepthwiseConv2D, Concatenate

def dl_model():
    input_layer = Input(shape=(32, 32, 3))
    
    # First Block: Dual-Path Structure
    main_path = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_layer)
    main_path = Conv2D(filters=128, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(main_path)
    main_path = Conv2D(filters=3, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(main_path)  
    branch_path = Conv2D(filters=3, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_layer)
    output_block1 = Add()([main_path, branch_path])

    # Second Block: Channel Splitting with Depthwise Separable Convolutions
    inputs_groups = Lambda(lambda x: tf.split(value=x, num_or_size_splits=3, axis=-1))(output_block1)
    conv1 = DepthwiseConv2D(kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(inputs_groups[0])
    conv2 = DepthwiseConv2D(kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(inputs_groups[1])
    conv3 = DepthwiseConv2D(kernel_size=(5, 5), strides=(1, 1), padding='same', activation='relu')(inputs_groups[2])
    output_block2 = Concatenate()([conv1, conv2, conv3])

    # Fully Connected Layers
    flatten = Flatten()(output_block2)
    dense1 = Dense(units=128, activation='relu')(flatten)
    output_layer = Dense(units=10, activation='softmax')(dense1)  

    model = keras.Model(inputs=input_layer, outputs=output_layer)
    return model