import keras
from keras.layers import Input, Conv2D, BatchNormalization, ReLU, Add, Flatten, Dense

def dl_model():
    input_layer = Input(shape=(32, 32, 3))
    
    # First block
    conv1 = Conv2D(filters=32, kernel_size=(3, 3), padding='same')(input_layer)
    bn1 = BatchNormalization()(conv1)
    act1 = ReLU()(bn1)
    
    # Second block
    conv2 = Conv2D(filters=64, kernel_size=(3, 3), padding='same')(act1)
    bn2 = BatchNormalization()(conv2)
    act2 = ReLU()(bn2)
    
    # Third block
    conv3 = Conv2D(filters=128, kernel_size=(3, 3), padding='same')(act2)
    bn3 = BatchNormalization()(conv3)
    act3 = ReLU()(bn3)
    
    # Direct parallel branch processing input
    parallel_conv = Conv2D(filters=128, kernel_size=(3, 3), padding='same')(input_layer)
    parallel_bn = BatchNormalization()(parallel_conv)
    parallel_act = ReLU()(parallel_bn)
    
    # Add outputs from all paths
    combined = Add()([act3, parallel_act])
    
    # Fully connected layers for classification
    flatten_layer = Flatten()(combined)
    dense1 = Dense(units=256, activation='relu')(flatten_layer)
    output_layer = Dense(units=10, activation='softmax')(dense1)
    
    model = keras.Model(inputs=input_layer, outputs=output_layer)
    
    return model