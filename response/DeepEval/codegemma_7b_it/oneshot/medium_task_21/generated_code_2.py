import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, BatchNormalization, Flatten, Dense, Dropout

def dl_model():     

    input_layer = Input(shape=(32, 32, 3))

    # Branch 1: 1x1 Convolution
    conv1_1x1 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_layer)

    # Branch 2: 1x1 followed by 3x3 Convolution
    conv2_1x1 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_layer)
    conv2_3x3 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(conv2_1x1)

    # Branch 3: 1x1 followed by two 3x3 Convolutions
    conv3_1x1 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_layer)
    conv3_3x3_1 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(conv3_1x1)
    conv3_3x3_2 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(conv3_3x3_1)

    # Branch 4: Average Pooling followed by 1x1 Convolution
    avg_pool = MaxPooling2D(pool_size=(2, 2), strides=2, padding='valid')(input_layer)
    conv4_1x1 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(avg_pool)

    # Concatenate outputs from all branches
    concat = Concatenate()([conv1_1x1, conv2_3x3, conv3_3x3_2, conv4_1x1])

    # Batch normalization and dropout
    bn = BatchNormalization()(concat)
    dropout = Dropout(rate=0.2)(bn)

    # Fully connected layers for classification
    flatten_layer = Flatten()(dropout)
    dense1 = Dense(units=512, activation='relu')(flatten_layer)
    dense2 = Dense(units=256, activation='relu')(dense1)
    output_layer = Dense(units=10, activation='softmax')(dense2)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model