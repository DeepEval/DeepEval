import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, BatchNormalization, Flatten, Dense, AveragePooling2D

def dl_model():
    input_layer = Input(shape=(28, 28, 1))
    
    # Block 1
    conv1 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_layer)
    avg_pool1_1x1 = AveragePooling2D(pool_size=(1, 1), strides=(1, 1))(conv1)
    avg_pool1_2x2 = AveragePooling2D(pool_size=(2, 2), strides=(2, 2))(conv1)
    avg_pool1_4x4 = AveragePooling2D(pool_size=(4, 4), strides=(4, 4))(conv1)
    
    flat1 = Flatten()(Concatenate()([avg_pool1_1x1, avg_pool1_2x2, avg_pool1_4x4]))
    
    # Connection to Block 2
    input_tensor = Flatten()(input_layer)
    
    # Block 2
    conv2 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_tensor)
    conv3 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_tensor)
    conv4 = Conv2D(filters=64, kernel_size=(5, 5), strides=(1, 1), padding='same', activation='relu')(input_tensor)
    avg_pool2 = AveragePooling2D(pool_size=(1, 1), strides=(1, 1))(conv4)
    
    avg_pool2_1x1 = AveragePooling2D(pool_size=(1, 1), strides=(1, 1))(conv3)
    avg_pool2_7x1 = AveragePooling2D(pool_size=(7, 1), strides=(1, 1))(conv3)
    avg_pool2_3x3 = AveragePooling2D(pool_size=(3, 3), strides=(1, 1))(conv3)
    flat2 = Concatenate()([avg_pool2_1x1, avg_pool2_7x1, avg_pool2_3x3, conv2])
    
    conv5 = Conv2D(filters=128, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(flat2)
    
    # Final layers
    batch_norm = BatchNormalization()(conv5)
    flatten = Flatten()(batch_norm)
    dense1 = Dense(units=256, activation='relu')(flatten)
    dense2 = Dense(units=128, activation='relu')(dense1)
    output_layer = Dense(units=10, activation='softmax')(dense2)
    
    model = keras.Model(inputs=input_layer, outputs=output_layer)
    return model

model = dl_model()
model.summary()