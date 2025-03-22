import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, BatchNormalization, Flatten, Dense, Dropout

def dl_model():
    # Input layer
    input_layer = Input(shape=(32, 32, 3))

    # Branch 1: 1x1 convolution
    conv1_1x1 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='valid', activation='relu')(input_layer)

    # Branch 2: 1x1 convolution followed by 3x3 convolution
    conv1_1x1_3x3 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='valid', activation='relu')(input_layer)
    conv2_3x3 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='valid', activation='relu')(conv1_1x1_3x3)

    # Branch 3: 1x1 convolution followed by two consecutive 3x3 convolutions
    conv1_1x1_2x3x3 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='valid', activation='relu')(input_layer)
    conv2_3x3 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='valid', activation='relu')(conv1_1x1_2x3x3)
    conv3_3x3 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='valid', activation='relu')(conv2_3x3)

    # Branch 4: Average pooling followed by 1x1 convolution
    avg_pool = MaxPooling2D(pool_size=(2, 2), strides=2, padding='valid')(input_layer)
    conv1_1x1_avg_pool = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='valid', activation='relu')(avg_pool)

    # Concatenate outputs from all branches
    concat = Concatenate()([conv1_1x1, conv1_1x1_3x3, conv1_1x1_2x3x3, conv1_1x1_avg_pool])

    # Batch normalization
    batch_norm = BatchNormalization()(concat)

    # Dropout
    dropout = Dropout(rate=0.25)(batch_norm)

    # Flatten
    flatten_layer = Flatten()(dropout)

    # Fully connected layers
    dense1 = Dense(units=512, activation='relu')(flatten_layer)
    dense2 = Dense(units=256, activation='relu')(dense1)
    dense3 = Dense(units=10, activation='softmax')(dense2)

    # Model definition
    model = keras.Model(inputs=input_layer, outputs=dense3)

    return model

# Build the model
model = dl_model()