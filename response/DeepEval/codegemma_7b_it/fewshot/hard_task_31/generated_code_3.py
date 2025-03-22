import keras
from keras.layers import Input, Conv2D, Dropout, MaxPooling2D, Add, Flatten, Dense, Lambda, tf

def dl_model():

    input_layer = Input(shape=(32, 32, 3))
    
    # Block 1
    def main_path(input_tensor):
        conv1 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        dropout1 = Dropout(rate=0.25)(conv1)
        conv2 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(dropout1)
        return conv2

    def branch_path(input_tensor):
        conv1 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        return conv1

    main_output = main_path(input_layer)
    branch_output = branch_path(input_layer)
    concat = Add()([main_output, branch_output])

    # Block 2
    def block2(input_tensor):
        inputs_groups = Lambda(lambda x: tf.split(value=x, num_or_size_splits=3, axis=-1))(input_tensor)
        conv1 = DepthwiseConv2D(kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(inputs_groups[0])
        conv1 = Dropout(rate=0.25)(conv1)
        conv2 = DepthwiseConv2D(kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(inputs_groups[1])
        conv2 = Dropout(rate=0.25)(conv2)
        conv3 = DepthwiseConv2D(kernel_size=(5, 5), strides=(1, 1), padding='same', activation='relu')(inputs_groups[2])
        conv3 = Dropout(rate=0.25)(conv3)
        concat = Concatenate()([conv1, conv2, conv3])
        return concat

    block2_output = block2(concat)

    # Output layer
    flatten = Flatten()(block2_output)
    output_layer = Dense(units=10, activation='softmax')(flatten)

    model = keras.Model(inputs=input_layer, outputs=output_layer)
    
    return model