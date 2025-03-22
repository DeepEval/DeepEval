import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, BatchNormalization, Flatten, Dense, Dropout

def dl_model():
    input_layer = Input(shape=(32, 32, 3))  # CIFAR-10 dataset has 32x32 images with 3 color channels

    # Branch 1
    conv1 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_layer)
    conv2 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(conv1)
    branch1 = Dropout(0.2)(conv2)  # Dropout layer to mitigate overfitting

    # Branch 2
    conv3 = Conv2D(filters=128, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_layer)
    conv4 = Conv2D(filters=128, kernel_size=(1, 7), strides=(1, 1), padding='same', activation='relu')(conv3)
    conv5 = Conv2D(filters=128, kernel_size=(7, 1), strides=(1, 1), padding='same', activation='relu')(conv4)
    conv6 = Conv2D(filters=128, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(conv5)
    branch2 = Dropout(0.2)(conv6)

    # Branch 3
    maxpool = MaxPooling2D(pool_size=(2, 2), strides=2, padding='same')(input_layer)
    branch3 = Dropout(0.2)(maxpool)

    # Concatenate the outputs from all branches
    output_tensor = Concatenate()([branch1, branch2, branch3])

    # Apply batch normalization and flatten the result
    batch_norm = BatchNormalization()(output_tensor)
    flatten_layer = Flatten()(batch_norm)

    # Process the output through three fully connected layers
    dense1 = Dense(units=128, activation='relu')(flatten_layer)
    dense2 = Dense(units=64, activation='relu')(dense1)
    output_layer = Dense(units=10, activation='softmax')(dense2)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model