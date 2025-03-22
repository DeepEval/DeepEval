import keras
from keras.layers import Input, Conv2D, Dropout, MaxPooling2D, Concatenate, DepthwiseConv2D, Flatten, Dense, Lambda

def dl_model():
    input_layer = Input(shape=(32, 32, 3))

    # Block 1
    conv_dropout_block = Conv2D(filters=64, kernel_size=(3, 3), activation='relu', padding='same')(input_layer)
    dropout1 = Dropout(0.2)(conv_dropout_block)
    restore_channels = Conv2D(filters=3, kernel_size=(1, 1), activation='relu', padding='same')(dropout1)

    branch_path = input_layer

    adding_layer = Add()([restore_channels, branch_path])

    # Block 2
    def split_and_process(input_tensor):
      split_tensor = Lambda(lambda x: tf.split(value=x, num_or_size_splits=3, axis=-1))(input_tensor)
      conv1 = DepthwiseConv2D(kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(split_tensor[0])
      dropout1 = Dropout(0.2)(conv1)
      conv2 = DepthwiseConv2D(kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(split_tensor[1])
      dropout2 = Dropout(0.2)(conv2)
      conv3 = DepthwiseConv2D(kernel_size=(5, 5), strides=(1, 1), padding='same', activation='relu')(split_tensor[2])
      dropout3 = Dropout(0.2)(conv3)
      return Concatenate()([dropout1, dropout2, dropout3])

    block2_output = split_and_process(adding_layer)

    flatten_layer = Flatten()(block2_output)
    output_layer = Dense(units=10, activation='softmax')(flatten_layer)

    model = keras.Model(inputs=input_layer, outputs=output_layer)
    return model