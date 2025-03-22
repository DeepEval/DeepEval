import keras
from keras.layers import Input, DepthwiseConv2D, BatchNormalization, Dense, Add
from keras.layers import GlobalAveragePooling2D, Reshape, Lambda

def dl_model():     

    input_layer = Input(shape=(32, 32, 3))
    
    # 7x7 depthwise separable convolutional layer
    depthwise_conv = DepthwiseConv2D(kernel_size=(7, 7), strides=(1, 1), padding='same', activation='relu')(input_layer)
    layer_norm = BatchNormalization()(depthwise_conv)
    
    # Global Average Pooling layer
    gap = GlobalAveragePooling2D()(layer_norm)
    
    # Reshape layer to match the original input shape
    reshape = Reshape((1, 1, 64))(gap)
    
    # Channel-wise feature transformation
    dense1 = Dense(units=64, activation='relu')(reshape)
    dense2 = Dense(units=64, activation='relu')(dense1)
    
    # Combine the original input with the processed features through an addition operation
    add = Add()([input_layer, dense2])
    
    # Global Average Pooling layer
    gap2 = GlobalAveragePooling2D()(add)
    
    # Final two fully connected layers
    dense3 = Dense(units=64, activation='relu')(gap2)
    output_layer = Dense(units=10, activation='softmax')(dense3)
    
    model = keras.Model(inputs=input_layer, outputs=output_layer)
    
    return model