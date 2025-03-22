import keras
from keras.layers import Input, Conv2D, Add, GlobalAveragePooling2D, Flatten, Dense, Dropout
from keras.regularizers import l2

def dl_model():
    
    input_layer = Input(shape=(28, 28, 1))
    
    # Main pathway
    conv_main = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_layer)
    conv1_main = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(conv_main)
    conv2_main = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(conv1_main)
    max_pool_main = MaxPooling2D(pool_size=(2, 2), strides=2, padding='same')(conv2_main)
    dropout_main = Dropout(0.5)(max_pool_main)

    # Branch pathway
    conv_branch = Conv2D(filters=32, kernel_size=(3, 3), strides=(2, 2), padding='same', activation='relu')(input_layer)
    conv_branch = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(conv_branch)

    # Fuse main and branch pathways
    add_layer = Add()([dropout_main, conv_branch])
    
    # Global average pooling, flatten, and dense layers
    gap = GlobalAveragePooling2D()(add_layer)
    flatten_layer = Flatten()(gap)
    dense1 = Dense(units=128, activation='relu', kernel_regularizer=l2(0.01))(flatten_layer)
    output_layer = Dense(units=10, activation='softmax')(dense1)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model