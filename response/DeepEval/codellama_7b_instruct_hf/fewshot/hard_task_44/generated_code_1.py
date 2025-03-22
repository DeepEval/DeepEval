import keras
from keras.layers import Input, Lambda, Dense, Flatten, Dropout, Concatenate, Conv2D, MaxPooling2D
from keras.models import Model

def dl_model():
    # Block 1: Feature Extraction
    inputs = Input(shape=(32, 32, 3))
    splits = Lambda(lambda x: tf.split(value=x, num_or_size_splits=3, axis=-1))(inputs)
    conv1 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(splits[0])
    conv2 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(splits[1])
    conv3 = Conv2D(filters=64, kernel_size=(5, 5), strides=(1, 1), padding='same', activation='relu')(splits[2])
    dropout = Dropout(rate=0.2)(concat([conv1, conv2, conv3]))
    # Block 2: Feature Fusion
    branch1 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(inputs)
    branch2 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(inputs)
    branch3 = Conv2D(filters=64, kernel_size=(5, 5), strides=(1, 1), padding='same', activation='relu')(inputs)
    branch4 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(inputs)
    branch5 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(branch4)
    branch6 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(branch5)
    branch7 = Conv2D(filters=64, kernel_size=(5, 5), strides=(1, 1), padding='same', activation='relu')(branch6)
    branch8 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(branch7)
    branch9 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(branch8)
    branch10 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(branch9)
    branch11 = Conv2D(filters=64, kernel_size=(5, 5), strides=(1, 1), padding='same', activation='relu')(branch10)
    branch12 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(branch11)
    branch13 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(branch12)
    branch14 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(branch13)
    branch15 = Conv2D(filters=64, kernel_size=(5, 5), strides=(1, 1), padding='same', activation='relu')(branch14)
    branch16 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(branch15)
    branch17 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(branch16)
    branch18 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(branch17)
    branch19 = Conv2D(filters=64, kernel_size=(5, 5), strides=(1, 1), padding='same', activation='relu')(branch18)
    branch20 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(branch19)
    output = Concatenate()([branch1, branch2, branch3, branch4, branch5, branch6, branch7, branch8, branch9, branch10, branch11, branch12, branch13, branch14, branch15, branch16, branch17, branch18, branch19, branch20])
    flatten = Flatten()(output)
    dense = Dense(units=10, activation='softmax')(flatten)
    model = Model(inputs=inputs, outputs=dense)
    return model