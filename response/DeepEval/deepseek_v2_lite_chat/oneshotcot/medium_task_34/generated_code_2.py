import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Dropout, Concatenate, BatchNormalization, Flatten, Dense, Conv2DTranspose
from keras.models import Model

def dl_model():
    # Input layer with shape (32, 32, 3) for CIFAR-10
    input_layer = Input(shape=(32, 32, 3))
    
    # Pass through 3 pairs of <convolutional layer, max-pooling layer>
    conv1 = Conv2D(filters=32, kernel_size=(3, 3), padding='same', activation='relu')(input_layer)
    pool1 = MaxPooling2D(pool_size=(2, 2), padding='same')(conv1)
    
    conv2 = Conv2D(filters=64, kernel_size=(3, 3), padding='same', activation='relu')(pool1)
    pool2 = MaxPooling2D(pool_size=(2, 2), padding='same')(conv2)
    
    conv3 = Conv2D(filters=128, kernel_size=(3, 3), padding='same', activation='relu')(pool2)
    pool3 = MaxPooling2D(pool_size=(2, 2), padding='same')(conv3)
    
    # Processing through <convolutional layer, Dropout layer, convolutional layer>
    drop1 = Dropout(0.2)(pool3)
    conv4 = Conv2D(filters=128, kernel_size=(3, 3), padding='same', activation='relu')(drop1)
    
    # Upsampling through 3 pairs of <convolutional layer, transposed convolutional layer>
    # With skip connections to the corresponding convolutional layers in the first part
    concat = Concatenate()([conv4])
    deconv1 = Conv2DTranspose(filters=128, kernel_size=(3, 3), strides=(2, 2), padding='same')(concat)
    bn1 = BatchNormalization()(deconv1)
    conv5 = Conv2D(filters=128, kernel_size=(3, 3), padding='same', activation='relu')(bn1)
    
    concat = Concatenate()([conv5, conv3])
    deconv2 = Conv2DTranspose(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same')(concat)
    bn2 = BatchNormalization()(deconv2)
    conv6 = Conv2D(filters=64, kernel_size=(3, 3), padding='same', activation='relu')(bn2)
    
    concat = Concatenate()([conv6, conv2])
    deconv3 = Conv2DTranspose(filters=32, kernel_size=(3, 3), strides=(2, 2), padding='same')(concat)
    bn3 = BatchNormalization()(deconv3)
    conv7 = Conv2D(filters=32, kernel_size=(3, 3), padding='same', activation='relu')(bn3)
    
    # Final 1x1 convolutional layer for probability output
    output = Conv2D(filters=10, kernel_size=(1, 1), padding='valid', activation='softmax')(conv7)
    
    model = Model(inputs=input_layer, outputs=output)
    
    return model

# Instantiate the model
model = dl_model()
model.summary()