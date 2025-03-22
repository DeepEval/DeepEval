import keras
from keras.layers import Input, Conv2D, MaxPooling2D, AvgPooling2D, Dropout, Flatten, Dense

def dl_model():
    # Input layer
    input_layer = Input(shape=(28, 28, 1))

    # Block 1: 3x3 convolution + 1x1 convolution + 1x1 convolution + average pooling
    conv1 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='valid', activation='relu')(input_layer)
    conv2 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='valid', activation='relu')(conv1)
    conv3 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='valid', activation='relu')(conv2)
    avg_pool = AvgPooling2D(pool_size=(2, 2), strides=2, padding='valid')(conv3)
    dropout = Dropout(rate=0.2)(avg_pool)

    # Block 2: 3x3 convolution + 1x1 convolution + 1x1 convolution + average pooling
    conv4 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='valid', activation='relu')(dropout)
    conv5 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='valid', activation='relu')(conv4)
    conv6 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='valid', activation='relu')(conv5)
    avg_pool2 = AvgPooling2D(pool_size=(2, 2), strides=2, padding='valid')(conv6)

    # Global average pooling
    global_pool = GlobalAveragePooling2D()(avg_pool2)

    # Flatten
    flatten = Flatten()(global_pool)

    # Fully connected layers
    dense1 = Dense(units=128, activation='relu')(flatten)
    dense2 = Dense(units=64, activation='relu')(dense1)
    output_layer = Dense(units=10, activation='softmax')(dense2)

    # Define the model
    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model