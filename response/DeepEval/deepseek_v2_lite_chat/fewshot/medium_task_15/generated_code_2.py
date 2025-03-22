import keras
from keras.layers import Input, Conv2D, BatchNormalization, Activation, GlobalAveragePooling2D, Dense, Reshape, Concatenate
from keras.models import Model

def dl_model():
    input_layer = Input(shape=(32, 32, 3))

    # Convolutional layer with batch normalization and ReLU activation
    conv1 = Conv2D(filters=32, kernel_size=(3, 3), padding='same')(input_layer)
    batchnorm1 = BatchNormalization()(conv1)
    relu1 = Activation('relu')(batchnorm1)

    # Global average pooling
    pool1 = GlobalAveragePooling2D()(relu1)

    # Fully connected layer 1
    dense1 = Dense(units=512, activation='relu')(pool1)

    # Fully connected layer 2
    dense2 = Dense(units=10, activation='softmax')(dense1)

    # Reshape dense1 to match the size of the initial feature maps
    reshape1 = Reshape((512,))(dense1)

    # Concatenate reshaped dense1 with the initial feature maps
    concat = Concatenate()([reshape1, relu1])

    # 1x1 convolution and average pooling to reduce dimensionality and downsample
    conv2 = Conv2D(filters=64, kernel_size=(1, 1), padding='same')(concat)
    batchnorm2 = BatchNormalization()(conv2)
    relu2 = Activation('relu')(batchnorm2)
    avg_pool = GlobalAveragePooling2D()(relu2)

    # Final fully connected layer
    output_layer = Dense(units=10, activation='softmax')(avg_pool)

    model = Model(inputs=input_layer, outputs=output_layer)

    return model

model = dl_model()
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])