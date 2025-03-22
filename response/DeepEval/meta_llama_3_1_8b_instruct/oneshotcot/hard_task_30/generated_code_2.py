import keras
from keras.layers import Input, Conv2D, Add, Lambda, DepthwiseConv2D, Concatenate, BatchNormalization, Flatten, Dense
from keras.regularizers import l2
from tensorflow.keras import backend as K
from tensorflow.keras import layers

def dl_model():

    input_layer = Input(shape=(32, 32, 3))

    # Define the first block
    def block1(input_tensor):
        path1 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu', kernel_regularizer=l2(0.01))(input_tensor)
        path1 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu', kernel_regularizer=l2(0.01))(path1)
        path1 = Conv2D(filters=3, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(path1)

        path2 = input_tensor

        output_tensor = Add()([path1, path2])

        return output_tensor

    block1_output = block1(input_layer)

    # Define the second block
    def block2(input_tensor):
        # Split the input tensor into three groups along the channel
        split_tensor = Lambda(lambda x: tf.split(x, num_or_size_splits=3, axis=-1))(input_tensor)
        
        group1 = DepthwiseConv2D(kernel_size=(1, 1), padding='same', activation='relu')(split_tensor[0])
        group2 = DepthwiseConv2D(kernel_size=(3, 3), padding='same', activation='relu')(split_tensor[1])
        group3 = DepthwiseConv2D(kernel_size=(5, 5), padding='same', activation='relu')(split_tensor[2])

        # Concatenate the outputs from the three groups
        output_tensor = Concatenate()([group1, group2, group3])

        return output_tensor

    block2_output = block2(block1_output)

    # Add batch normalization and flatten the result
    bath_norm = BatchNormalization()(block2_output)
    flatten_layer = Flatten()(bath_norm)

    # Add dense layers to produce the final classification
    dense1 = Dense(units=128, activation='relu')(flatten_layer)
    output_layer = Dense(units=10, activation='softmax')(dense1)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model