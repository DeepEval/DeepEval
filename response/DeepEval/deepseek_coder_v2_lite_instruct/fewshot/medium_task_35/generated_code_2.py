import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Add, Flatten, Dense, Dropout, UpSampling2D, Concatenate

def dl_model():
    input_layer = Input(shape=(32, 32, 3))

    # Stage 1: Convolution and Max Pooling
    conv1 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_layer)
    pool1 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(conv1)
    
    conv2 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(pool1)
    pool2 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(conv2)
    
    # Stage 2: Additional Convolutional and Dropout Layers
    conv3 = Conv2D(filters=128, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(pool2)
    dropout1 = Dropout(0.5)(conv3)
    
    conv4 = Conv2D(filters=128, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(dropout1)
    dropout2 = Dropout(0.5)(conv4)
    
    # Skip Connections
    upsample1 = UpSampling2D(size=(2, 2))(dropout2)
    concat1 = Concatenate()([conv2, upsample1])
    
    conv5 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(concat1)
    upsample2 = UpSampling2D(size=(2, 2))(conv5)
    concat2 = Concatenate()([conv1, upsample2])
    
    conv6 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(concat2)
    
    # Final 1x1 Convolutional Layer
    final_conv = Conv2D(filters=10, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='softmax')(conv6)
    
    # Flatten and Dense Layers (Optional for CIFAR-10)
    flatten_layer = Flatten()(final_conv)
    output_layer = Dense(units=10, activation='softmax')(flatten_layer)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model