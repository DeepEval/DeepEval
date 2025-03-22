import keras
from keras.layers import Input, Conv2D, GlobalAveragePooling2D, GlobalMaxPooling2D, Concatenate, Dense, Activation, BatchNormalization, Flatten

def dl_model():
    input_layer = Input(shape=(32, 32, 3))
    conv_layer = Conv2D(filters=32, kernel_size=(3, 3), activation='relu')(input_layer)
    pool1 = GlobalAveragePooling2D()(conv_layer)
    dense1 = Dense(units=128, activation='relu')(pool1)
    dense2 = Dense(units=64, activation='relu')(dense1)

    # Global max pooling path
    conv_layer = Conv2D(filters=32, kernel_size=(3, 3), activation='relu')(input_layer)
    pool2 = GlobalMaxPooling2D()(conv_layer)
    dense3 = Dense(units=128, activation='relu')(pool2)
    dense4 = Dense(units=64, activation='relu')(dense3)

    # Concatenate the outputs from both paths
    concat = Concatenate()( [pool1, pool2] )
    attention = Dense(units=2, activation='softmax')(concat)  # 2 classes for channel attention weights

    # Element-wise multiplication of features with attention weights
    att_pool1 = Activation('mul')( [dense1, attention] )
    att_pool2 = Activation('mul')( [dense3, attention] )

    # Average and max pooling paths
    avg_pool = BatchNormalization()(GlobalAveragePooling2D()(conv_layer))
    avg_pool = Flatten()(avg_pool)
    avg_dense1 = Dense(units=128, activation='relu')(avg_pool)
    avg_dense2 = Dense(units=64, activation='relu')(avg_dense1)

    max_pool = BatchNormalization()(GlobalMaxPooling2D()(conv_layer))
    max_pool = Flatten()(max_pool)
    max_dense1 = Dense(units=128, activation='relu')(max_pool)
    max_dense2 = Dense(units=64, activation='relu')(max_dense1)

    # Concatenate the outputs from both pooling paths
    concat_pool = Concatenate()([att_pool1, att_pool2, avg_pool, max_pool])
    output_layer = Dense(units=10, activation='softmax')(concat_pool)

    model = keras.Model(inputs=input_layer, outputs=output_layer)
    return model