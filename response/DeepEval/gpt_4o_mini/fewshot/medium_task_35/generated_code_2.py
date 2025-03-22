import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Dropout, UpSampling2D, Add, Flatten, Dense

def dl_model():
    input_layer = Input(shape=(32, 32, 3))

    # Stage 1: Convolutions and Max Pooling
    conv1 = Conv2D(filters=32, kernel_size=(3, 3), padding='same', activation='relu')(input_layer)
    conv1 = Conv2D(filters=32, kernel_size=(3, 3), padding='same', activation='relu')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

    # Stage 2: More Convolutions and Max Pooling
    conv2 = Conv2D(filters=64, kernel_size=(3, 3), padding='same', activation='relu')(pool1)
    conv2 = Conv2D(filters=64, kernel_size=(3, 3), padding='same', activation='relu')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

    # Additional Convolutions and Dropout
    conv3 = Conv2D(filters=128, kernel_size=(3, 3), padding='same', activation='relu')(pool2)
    conv3 = Conv2D(filters=128, kernel_size=(3, 3), padding='same', activation='relu')(conv3)
    dropout1 = Dropout(0.5)(conv3)

    # Upsampling with Skip Connections
    upsampling1 = UpSampling2D(size=(2, 2))(dropout1)
    skip1 = Add()([upsampling1, conv2])  # Skip connection
    conv4 = Conv2D(filters=64, kernel_size=(3, 3), padding='same', activation='relu')(skip1)
    conv4 = Conv2D(filters=64, kernel_size=(3, 3), padding='same', activation='relu')(conv4)

    upsampling2 = UpSampling2D(size=(2, 2))(conv4)
    skip2 = Add()([upsampling2, conv1])  # Skip connection
    conv5 = Conv2D(filters=32, kernel_size=(3, 3), padding='same', activation='relu')(skip2)
    conv5 = Conv2D(filters=32, kernel_size=(3, 3), padding='same', activation='relu')(conv5)

    # Final Convolution to produce probability outputs
    output_layer = Conv2D(filters=10, kernel_size=(1, 1), activation='softmax')(conv5)

    # Build the model
    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model