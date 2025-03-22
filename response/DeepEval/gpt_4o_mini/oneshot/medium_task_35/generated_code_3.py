import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Dropout, UpSampling2D, Concatenate, BatchNormalization, Flatten, Dense

def dl_model():
    input_layer = Input(shape=(32, 32, 3))  # CIFAR-10 images are 32x32 pixels with 3 color channels (RGB)

    # Stage 1: Convolution and Max Pooling
    conv1 = Conv2D(filters=32, kernel_size=(3, 3), padding='same', activation='relu')(input_layer)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
    dropout1 = Dropout(0.25)(pool1)

    # Stage 2: Convolution and Max Pooling
    conv2 = Conv2D(filters=64, kernel_size=(3, 3), padding='same', activation='relu')(dropout1)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
    dropout2 = Dropout(0.25)(pool2)

    # Additional Convolutions and Dropout
    conv3 = Conv2D(filters=128, kernel_size=(3, 3), padding='same', activation='relu')(dropout2)
    dropout3 = Dropout(0.25)(conv3)

    # UpSampling and Skip Connection 1
    upsample1 = UpSampling2D(size=(2, 2))(dropout3)
    concat1 = Concatenate()([upsample1, conv2])  # Skip connection from conv2
    conv4 = Conv2D(filters=64, kernel_size=(3, 3), padding='same', activation='relu')(concat1)

    # UpSampling and Skip Connection 2
    upsample2 = UpSampling2D(size=(2, 2))(conv4)
    concat2 = Concatenate()([upsample2, conv1])  # Skip connection from conv1
    conv5 = Conv2D(filters=32, kernel_size=(3, 3), padding='same', activation='relu')(concat2)

    # Final 1x1 Convolution to produce probability outputs
    output_layer = Conv2D(filters=10, kernel_size=(1, 1), activation='softmax')(conv5)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model