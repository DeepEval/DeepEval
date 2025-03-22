import keras
from keras.layers import Input, Conv2D, AveragePooling2D, Flatten, Concatenate, Dense, Reshape, Dropout

def dl_model():     

    input_layer = Input(shape=(28, 28, 1))
    
    # Block 1: Average Pooling
    pool1 = AveragePooling2D(pool_size=(1, 1), strides=(1, 1))(input_layer)
    pool2 = AveragePooling2D(pool_size=(2, 2), strides=(2, 2))(input_layer)
    pool4 = AveragePooling2D(pool_size=(4, 4), strides=(4, 4))(input_layer)

    flat1 = Flatten()(pool1)
    flat2 = Flatten()(pool2)
    flat4 = Flatten()(pool4)
    
    concat_block1 = Concatenate()([flat1, flat2, flat4])
    reshape_layer = Reshape((1, 1, 3))(concat_block1)
    dense1 = Dense(units=128, activation='relu')(reshape_layer)

    # Block 2: Feature Extraction Paths
    conv1_1 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(dense1)
    dropout1 = Dropout(0.2)(conv1_1)
    
    conv2_1 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(dense1)
    conv2_2 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(conv2_1)
    conv2_3 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(conv2_2)
    dropout2 = Dropout(0.2)(conv2_3)
    
    conv3_1 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(dense1)
    conv3_2 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(conv3_1)
    dropout3 = Dropout(0.2)(conv3_2)
    
    pool4_1 = AveragePooling2D(pool_size=(2, 2), strides=(2, 2))(dense1)
    conv4_1 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(pool4_1)
    dropout4 = Dropout(0.2)(conv4_1)

    concat_block2 = Concatenate(axis=3)([dropout1, dropout2, dropout3, dropout4])
    
    # Output Layer
    flatten_layer = Flatten()(concat_block2)
    dense2 = Dense(units=64, activation='relu')(flatten_layer)
    output_layer = Dense(units=10, activation='softmax')(dense2)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model