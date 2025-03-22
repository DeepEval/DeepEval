import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Dropout, concatenate, Flatten, Dense

def dl_model():

    # Pathway 1
    input_layer = Input(shape=(28, 28, 1))
    conv1 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_layer)
    conv1_1 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(conv1)
    maxpool1 = MaxPooling2D(pool_size=(2, 2), strides=(1, 1), padding='same')(conv1_1)
    dropout1 = Dropout(0.5)(maxpool1)

    # Pathway 2
    branch_input = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_layer)
    branch_conv1 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(branch_input)
    maxpool2 = MaxPooling2D(pool_size=(2, 2), strides=(1, 1), padding='same')(branch_conv1)

    # Fusion
    concat_layer = concatenate([maxpool1, maxpool2])
    conv2 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(concat_layer)

    # Global average pooling, flattening and fully connected layer
    avg_pool = GlobalAvgPool2D()(conv2)
    flatten = Flatten()(avg_pool)
    dense = Dense(units=128, activation='relu')(flatten)
    output = Dense(units=10, activation='softmax')(dense)

    model = keras.Model(inputs=input_layer, outputs=output)

    return model