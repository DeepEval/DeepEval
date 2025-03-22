import keras
from keras.layers import Input, Conv2D, MaxPooling2D, GlobalAveragePooling2D, Dense, Reshape, Add

def dl_model():     
    input_layer = Input(shape=(32, 32, 3))

    # Main path
    conv1 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_layer)
    conv2 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(conv1)
    conv3 = Conv2D(filters=128, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(conv2)
    main_path = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(conv3)

    # Branch path
    avg_pool = GlobalAveragePooling2D()(input_layer)
    dense1 = Dense(units=128, activation='relu')(avg_pool)
    dense2 = Dense(units=128, activation='relu')(dense1)
    branch_weights = Reshape((32*32,))(dense2)  

    # Element-wise multiplication
    branch_output = input_layer * branch_weights

    # Add outputs and further classification
    merged = Add()([main_path, branch_output])
    dense3 = Dense(units=256, activation='relu')(merged)
    output_layer = Dense(units=10, activation='softmax')(dense3)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model