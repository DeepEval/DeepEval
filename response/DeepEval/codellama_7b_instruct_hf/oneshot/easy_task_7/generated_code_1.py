import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Dropout, BatchNormalization, Flatten, Dense

def dl_model():
    # Main path
    input_layer = Input(shape=(28, 28, 1))
    conv1 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_layer)
    conv2 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(conv1)
    dropout = Dropout(rate=0.2)(conv2)
    conv3 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(dropout)
    output_main = conv3

    # Branch path
    branch_input = Input(shape=(28, 28, 1))
    conv1_branch = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(branch_input)
    conv2_branch = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(conv1_branch)
    output_branch = conv2_branch

    # Combine main and branch paths
    output_merge = keras.layers.Add()([output_main, output_branch])

    # Flatten and fully connected layers
    flatten = Flatten()(output_merge)
    dense = Dense(units=128, activation='relu')(flatten)
    output_layer = Dense(units=10, activation='softmax')(dense)

    model = keras.Model(inputs=[input_layer, branch_input], outputs=output_layer)
    return model