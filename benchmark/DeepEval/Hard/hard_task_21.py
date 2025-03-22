import tensorflow as tf 
from keras.models import Model
from keras.layers import Input, Lambda, DepthwiseConv2D, Concatenate,  Conv2D, Add, Dense

def dl_model():

    input_layer = Input(shape=(32, 32, 3))

    inputs_groups = Lambda(function=lambda x: tf.split(value=x, num_or_size_splits=3, axis=-1))(input_layer)
    conv1 = DepthwiseConv2D(kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(inputs_groups[0])
    conv2 = DepthwiseConv2D(kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(inputs_groups[1])
    conv3 = DepthwiseConv2D(kernel_size=(5, 5), strides=(1, 1), padding='same', activation='relu')(inputs_groups[2])
    main_path = Concatenate()([conv1, conv2, conv3])

    branch_path = Conv2D(filters=main_path.shape[-1], kernel_size=(1, 1), strides=(1, 1), padding='same')(input_layer)
    added = Add()([main_path, branch_path])
    
    dense_1_output = Dense(units=128, activation='relu')(added)
    output_layer = Dense(units=10, activation='softmax')(dense_1_output)

    model = Model(inputs=input_layer, outputs=output_layer)

    return model
