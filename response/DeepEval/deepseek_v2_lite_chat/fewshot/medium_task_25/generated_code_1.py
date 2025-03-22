import keras
from keras.layers import Input, Conv2D, MaxPooling2D, ZeroPadding2D, Concatenate, Dense, Flatten
from keras.models import Model

def dl_model():
    input_layer = Input(shape=(32, 32, 3))

    # Path 1: Single 1x1 convolution
    conv1_1 = Conv2D(filters=32, kernel_size=1, activation='relu')(input_layer)

    # Path 2: Average pooling followed by a 1x1 convolution
    avg_pool = MaxPooling2D(pool_size=(2, 2))(input_layer)
    conv1_2 = Conv2D(filters=32, kernel_size=1, activation='relu')(avg_pool)

    # Path 3: 1x1 convolution followed by two parallel 1x3 and 3x1 convolutions
    conv1_3 = Conv2D(filters=32, kernel_size=1, activation='relu')(input_layer)
    conv1_4 = Conv2D(filters=32, kernel_size=3, padding='same', activation='relu')(input_layer)
    conv1_5 = Conv2D(filters=32, kernel_size=3, padding='same', activation='relu')(input_layer)
    concat1 = Concatenate(axis=-1)([conv1_3, conv1_4, conv1_5])

    # Path 4: 1x1 convolution followed by a 3x3 convolution
    conv1_6 = Conv2D(filters=32, kernel_size=1, activation='relu')(input_layer)
    conv1_7 = Conv2D(filters=32, kernel_size=3, padding='same', activation='relu')(input_layer)
    conv1_8 = Conv2D(filters=32, kernel_size=3, padding='same', activation='relu')(input_layer)
    concat2 = Concatenate(axis=-1)([conv1_6, conv1_7, conv1_8])

    # Concatenate the results from all paths
    concat = Concatenate(axis=-1)([concat1, concat2])

    # Fully connected layer for classification
    output_layer = Dense(units=10, activation='softmax')(concat)

    # Construct the model
    model = Model(inputs=input_layer, outputs=output_layer)

    return model