import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Add, Flatten, Dense, GlobalAveragePooling2D, Reshape, Multiply

def dl_model():

    input_layer = Input(shape=(32, 32, 3))
    
    # Main path
    conv1_1 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_layer)
    conv1_2 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(conv1_1)
    conv1_3 = Conv2D(filters=128, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(conv1_2)
    max_pool_main = MaxPooling2D(pool_size=(2, 2), strides=2, padding='valid')(conv1_3)

    # Branch path
    avg_pool_branch = GlobalAveragePooling2D()(input_layer)
    dense_branch_1 = Dense(units=128, activation='relu')(avg_pool_branch)
    dense_branch_2 = Dense(units=128, activation='relu')(dense_branch_1)

    # Reshape and multiply weights
    reshape_dense_branch = Reshape(target_shape=(1, 1, 128))(dense_branch_2)
    multiply_weights = Multiply()([reshape_dense_branch, max_pool_main])

    # Add outputs from both paths
    add_layers = Add()([multiply_weights, max_pool_main])

    # Additional fully connected layers for classification
    dense_2 = Dense(units=128, activation='relu')(add_layers)
    dense_3 = Dense(units=10, activation='softmax')(dense_2)

    model = keras.Model(inputs=input_layer, outputs=dense_3)

    return model