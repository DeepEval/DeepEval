import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, BatchNormalization, Flatten, Dense
from keras.models import Model
from keras.applications.vgg16 import VGG16

def dl_model():
    input_layer = Input(shape=(32, 32, 3))

    # 3x3 convolution branch
    conv1 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_layer)
    conv2 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(conv1)
    branch1 = Concatenate()([conv1, conv2])

    # 1x1 convolution branch
    conv1 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_layer)
    conv2 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(conv1)
    conv3 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(conv2)
    branch2 = Concatenate()([conv1, conv2, conv3])

    # Max pooling branch
    maxpool = MaxPooling2D(pool_size=(2, 2), strides=2, padding='same')(input_layer)
    branch3 = Concatenate()([maxpool, maxpool])

    # Multi-scale feature fusion block
    fused_features = Concatenate()([branch1, branch2, branch3])

    # Flatten and pass through two fully connected layers for classification
    flat = Flatten()(fused_features)
    dense1 = Dense(units=128, activation='relu')(flat)
    dense2 = Dense(units=64, activation='relu')(dense1)
    output_layer = Dense(units=10, activation='softmax')(dense2)

    model = Model(inputs=input_layer, outputs=output_layer)

    return model