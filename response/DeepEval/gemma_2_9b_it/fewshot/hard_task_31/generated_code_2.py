import keras
from keras.layers import Input, Conv2D, Dropout, MaxPooling2D, Concatenate, DepthwiseConv2D, Flatten, Dense, Lambda

def dl_model():
    input_layer = Input(shape=(32, 32, 3))

    # Block 1
    def conv_dropout_block(input_tensor):
        x = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        x = Dropout(0.2)(x)
        x = Conv2D(filters=3, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(x) 
        return x

    main_path = conv_dropout_block(input_layer)
    branch_path = input_layer
    adding_layer = keras.layers.Add()([main_path, branch_path])

    # Block 2
    def separable_conv_block(input_tensor):
        inputs_groups = Lambda(lambda x: tf.split(value=x, num_or_size_splits=3, axis=-1))(input_tensor)
        conv1 = DepthwiseConv2D(kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(inputs_groups[0])
        conv1 = Dropout(0.2)(conv1)
        conv2 = DepthwiseConv2D(kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(inputs_groups[1])
        conv2 = Dropout(0.2)(conv2)
        conv3 = DepthwiseConv2D(kernel_size=(5, 5), strides=(1, 1), padding='same', activation='relu')(inputs_groups[2])
        conv3 = Dropout(0.2)(conv3)
        return Concatenate()([conv1, conv2, conv3])

    block2_output = separable_conv_block(adding_layer)

    flatten_layer = Flatten()(block2_output)
    output_layer = Dense(units=10, activation='softmax')(flatten_layer)
    
    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model