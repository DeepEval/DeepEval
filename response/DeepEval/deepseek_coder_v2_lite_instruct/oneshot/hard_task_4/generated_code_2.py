import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, BatchNormalization, Flatten, Dense, GlobalAveragePooling2D, Reshape, Multiply
from keras.models import Model

def dl_model():
    input_layer = Input(shape=(32, 32, 3))

    # Initial 1x1 convolution and 3x3 depthwise separable convolution
    conv1 = Conv2D(filters=3, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_layer)
    depthwise_conv = Conv2D(filters=3, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu', depthwise_mode=True)(conv1)

    # Channel attention mechanism
    gap = GlobalAveragePooling2D()(depthwise_conv)
    dense1 = Dense(units=depthwise_conv.shape[-1]//16, activation='relu')(gap)
    dense2 = Dense(units=depthwise_conv.shape[-1], activation='sigmoid')(dense1)
    reshape = Reshape((1, 1, depthwise_conv.shape[-1]))(dense2)
    multiply = Multiply()([depthwise_conv, reshape])

    # Concatenate initial input with weighted features
    combined = Concatenate()([input_layer, multiply])

    # Flatten and fully connected layers
    flatten = Flatten()(combined)
    dense3 = Dense(units=256, activation='relu')(flatten)
    output_layer = Dense(units=10, activation='softmax')(dense3)

    model = Model(inputs=input_layer, outputs=output_layer)
    return model

# Create and compile the model
model = dl_model()
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])