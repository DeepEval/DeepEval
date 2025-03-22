import keras
from keras.layers import Input, Conv2D, BatchNormalization, ReLU, Add, Flatten, Dense

def dl_model():
    input_layer = Input(shape=(32, 32, 3))

    # Sequential block 1
    conv1 = Conv2D(filters=32, kernel_size=(3, 3), padding='same')(input_layer)
    bn1 = BatchNormalization()(conv1)
    relu1 = ReLU()(bn1)
    
    # Sequential block 2
    conv2 = Conv2D(filters=64, kernel_size=(3, 3), padding='same')(relu1)
    bn2 = BatchNormalization()(conv2)
    relu2 = ReLU()(bn2)
    
    # Sequential block 3
    conv3 = Conv2D(filters=128, kernel_size=(3, 3), padding='same')(relu2)
    bn3 = BatchNormalization()(conv3)
    relu3 = ReLU()(bn3)

    # Parallel branch processing the input directly
    parallel_conv = Conv2D(filters=128, kernel_size=(3, 3), padding='same')(input_layer)
    parallel_bn = BatchNormalization()(parallel_conv)
    parallel_relu = ReLU()(parallel_bn)

    # Add outputs from the sequential blocks and the parallel branch
    adding_layer = Add()([relu1, relu2, relu3, parallel_relu])

    # Fully connected layers for classification
    flatten_layer = Flatten()(adding_layer)
    dense1 = Dense(units=128, activation='relu')(flatten_layer)
    output_layer = Dense(units=10, activation='softmax')(dense1)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model