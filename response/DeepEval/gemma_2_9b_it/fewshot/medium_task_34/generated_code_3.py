import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Dropout, UpSampling2D, Conv2DTranspose, Add

def dl_model():
    
    input_layer = Input(shape=(32, 32, 3))

    # Part 1: Feature Extraction
    conv1 = Conv2D(filters=32, kernel_size=(3, 3), activation='relu')(input_layer)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
    conv2 = Conv2D(filters=64, kernel_size=(3, 3), activation='relu')(pool1)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
    conv3 = Conv2D(filters=128, kernel_size=(3, 3), activation='relu')(pool2)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

    # Part 2: Generalization Enhancement
    conv4 = Conv2D(filters=128, kernel_size=(3, 3), activation='relu')(pool3)
    dropout = Dropout(rate=0.5)(conv4)
    conv5 = Conv2D(filters=128, kernel_size=(3, 3), activation='relu')(dropout)

    # Part 3: Upsampling and Reconstruction
    up6 = Conv2DTranspose(filters=128, kernel_size=(2, 2), strides=2)(conv5)
    merge6 = Add()([up6, conv3])
    conv6 = Conv2D(filters=128, kernel_size=(3, 3), activation='relu')(merge6)
    up7 = Conv2DTranspose(filters=64, kernel_size=(2, 2), strides=2)(conv6)
    merge7 = Add()([up7, conv2])
    conv7 = Conv2D(filters=64, kernel_size=(3, 3), activation='relu')(merge7)
    up8 = Conv2DTranspose(filters=32, kernel_size=(2, 2), strides=2)(conv7)
    merge8 = Add()([up8, conv1])
    conv8 = Conv2D(filters=32, kernel_size=(3, 3), activation='relu')(merge8)

    # Final Classification
    output_layer = Conv2D(filters=10, kernel_size=(1, 1), activation='softmax')(conv8)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model