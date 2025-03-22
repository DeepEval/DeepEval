import keras
from keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, Conv2DTranspose, Concatenate, Dropout, BatchNormalization, Flatten, Dense
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint
from keras.preprocessing.image import ImageDataGenerator

def dl_model():
    
    input_layer = Input(shape=(32, 32, 3))
    
    # Feature extraction
    conv1 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_layer)
    pool1 = MaxPooling2D(pool_size=(2, 2), strides=2, padding='same')(conv1)

    conv2 = Conv2D(filters=128, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(pool1)
    pool2 = MaxPooling2D(pool_size=(2, 2), strides=2, padding='same')(conv2)

    conv3 = Conv2D(filters=256, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(pool2)
    pool3 = MaxPooling2D(pool_size=(2, 2), strides=2, padding='same')(conv3)

    # Generalization enhancement
    conv4 = Conv2D(filters=256, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(pool3)
    drop = Dropout(0.2)(conv4)
    conv5 = Conv2D(filters=256, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(drop)

    # Spatial information restoration
    up1 = Conv2DTranspose(filters=256, kernel_size=(2, 2), strides=(2, 2), padding='same')(conv5)
    skip1 = Concatenate()([up1, conv3])
    conv6 = Conv2D(filters=256, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(skip1)

    up2 = Conv2DTranspose(filters=128, kernel_size=(2, 2), strides=(2, 2), padding='same')(conv6)
    skip2 = Concatenate()([up2, conv2])
    conv7 = Conv2D(filters=128, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(skip2)

    up3 = Conv2DTranspose(filters=64, kernel_size=(2, 2), strides=(2, 2), padding='same')(conv7)
    skip3 = Concatenate()([up3, conv1])
    conv8 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(skip3)

    # Output
    output = Conv2D(filters=10, kernel_size=(1, 1), strides=(1, 1), padding='same')(conv8)
    output = Flatten()(output)
    output = Dense(units=10, activation='softmax')(output)

    model = keras.Model(inputs=input_layer, outputs=output)

    return model

# Compile the model
model = dl_model()
model.compile(optimizer=Adam(lr=0.001), loss='categorical_crossentropy', metrics=['accuracy'])