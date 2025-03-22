import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Add, Concatenate, UpSampling2D, Dropout, Dense

def dl_model():
    # Define the input shape of the model
    input_layer = Input(shape=(32, 32, 3))

    # Stage 1: Downsampling using convolution and max pooling
    conv1 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_layer)
    pool1 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(conv1)

    conv2 = Conv2D(filters=128, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(pool1)
    pool2 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(conv2)

    # Stage 2: Additional convolution and dropout
    conv3 = Conv2D(filters=256, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(pool2)
    dropout = Dropout(0.2)(conv3)

    # Upsampling using skip connections
    up1 = UpSampling2D(size=(2, 2))(dropout)
    conv4 = Conv2D(filters=128, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(up1)
    skip_connection1 = Add()([conv4, conv2])
    conv5 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(skip_connection1)
    up2 = UpSampling2D(size=(2, 2))(conv5)
    skip_connection2 = Add()([up2, conv1])

    # Final stage: 1x1 convolution for classification
    conv6 = Conv2D(filters=10, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='softmax')(skip_connection2)

    # Flatten the output to produce the final classification result
    output_layer = Flatten()(conv6)

    # Create the model
    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model