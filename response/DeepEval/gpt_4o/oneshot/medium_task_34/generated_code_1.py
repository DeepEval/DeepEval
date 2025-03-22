import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Dropout, Conv2DTranspose, Add
from keras.models import Model

def dl_model():
    input_layer = Input(shape=(32, 32, 3))
    
    # Part 1: Feature extraction using 3 pairs of <Convolutional Layer, Max-Pooling Layer>
    conv1 = Conv2D(filters=32, kernel_size=(3, 3), padding='same', activation='relu')(input_layer)
    pool1 = MaxPooling2D(pool_size=(2, 2), strides=2, padding='same')(conv1)

    conv2 = Conv2D(filters=64, kernel_size=(3, 3), padding='same', activation='relu')(pool1)
    pool2 = MaxPooling2D(pool_size=(2, 2), strides=2, padding='same')(conv2)

    conv3 = Conv2D(filters=128, kernel_size=(3, 3), padding='same', activation='relu')(pool2)
    pool3 = MaxPooling2D(pool_size=(2, 2), strides=2, padding='same')(conv3)
    
    # Part 2: Generalization using <Convolutional Layer, Dropout Layer, Convolutional Layer>
    conv4 = Conv2D(filters=256, kernel_size=(3, 3), padding='same', activation='relu')(pool3)
    dropout = Dropout(rate=0.5)(conv4)
    conv5 = Conv2D(filters=256, kernel_size=(3, 3), padding='same', activation='relu')(dropout)

    # Part 3: Upsampling using 3 pairs of <Convolutional Layer, Transposed Convolutional Layer> with skip connections
    conv6 = Conv2D(filters=128, kernel_size=(3, 3), padding='same', activation='relu')(conv5)
    up1 = Conv2DTranspose(filters=128, kernel_size=(2, 2), strides=2, padding='same', activation='relu')(conv6)
    skip1 = Add()([up1, conv3])  # Skip connection

    conv7 = Conv2D(filters=64, kernel_size=(3, 3), padding='same', activation='relu')(skip1)
    up2 = Conv2DTranspose(filters=64, kernel_size=(2, 2), strides=2, padding='same', activation='relu')(conv7)
    skip2 = Add()([up2, conv2])  # Skip connection

    conv8 = Conv2D(filters=32, kernel_size=(3, 3), padding='same', activation='relu')(skip2)
    up3 = Conv2DTranspose(filters=32, kernel_size=(2, 2), strides=2, padding='same', activation='relu')(conv8)
    skip3 = Add()([up3, conv1])  # Skip connection

    # Final 1x1 Convolution to generate class probability output
    output_layer = Conv2D(filters=10, kernel_size=(1, 1), activation='softmax')(skip3)

    # Define the model
    model = Model(inputs=input_layer, outputs=output_layer)

    return model