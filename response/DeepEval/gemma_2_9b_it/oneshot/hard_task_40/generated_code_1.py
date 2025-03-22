import keras
from keras.layers import Input, Conv2D, AveragePooling2D, Flatten, Concatenate, Dense, Reshape, Dropout, BatchNormalization

def dl_model(): 
    input_layer = Input(shape=(28, 28, 1))

    # Block 1: Average Pooling and Concatenation
    pool1 = AveragePooling2D(pool_size=(1, 1), strides=1)(input_layer)
    pool2 = AveragePooling2D(pool_size=(2, 2), strides=2)(input_layer)
    pool3 = AveragePooling2D(pool_size=(4, 4), strides=4)(input_layer)

    flat1 = Flatten()(pool1)
    flat2 = Flatten()(pool2)
    flat3 = Flatten()(pool3)
    concat_output = Concatenate()([flat1, flat2, flat3])
    reshape_layer = Reshape((1, 1, 9))(concat_output) 

    dense1 = Dense(units=128, activation='relu')(reshape_layer)
    

    # Block 2: Multi-Scale Feature Extraction
    conv1_1 = Conv2D(filters=64, kernel_size=(1, 1), strides=1, padding='same', activation='relu')(dense1)
    
    conv2_1 = Conv2D(filters=64, kernel_size=(1, 1), strides=1, padding='same', activation='relu')(dense1)
    conv2_2 = Conv2D(filters=64, kernel_size=(3, 3), strides=1, padding='same', activation='relu')(conv2_1)
    conv2_3 = Conv2D(filters=64, kernel_size=(3, 3), strides=1, padding='same', activation='relu')(conv2_2)

    conv3_1 = Conv2D(filters=64, kernel_size=(1, 1), strides=1, padding='same', activation='relu')(dense1)
    conv3_2 = Conv2D(filters=64, kernel_size=(3, 3), strides=1, padding='same', activation='relu')(conv3_1)

    pool4_1 = AveragePooling2D(pool_size=(2, 2), strides=2, padding='same')(dense1)
    pool4_2 = Conv2D(filters=64, kernel_size=(1, 1), strides=1, padding='same', activation='relu')(pool4_1)

    drop1 = Dropout(0.5)(conv1_1)
    drop2 = Dropout(0.5)(conv2_3)
    drop3 = Dropout(0.5)(conv3_2)
    drop4 = Dropout(0.5)(pool4_2)

    concat_output_block2 = Concatenate()([drop1, drop2, drop3, drop4])
    
    dense2 = Dense(units=64, activation='relu')(concat_output_block2)
    output_layer = Dense(units=10, activation='softmax')(dense2)

    model = keras.Model(inputs=input_layer, outputs=output_layer)
    return model