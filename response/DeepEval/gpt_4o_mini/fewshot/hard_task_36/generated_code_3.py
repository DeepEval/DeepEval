import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Dropout, Add, GlobalAveragePooling2D, Flatten, Dense

def dl_model():
    input_layer = Input(shape=(28, 28, 1))

    # Main pathway
    conv3x3 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_layer)
    
    conv1x1_1 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(conv3x3)
    conv1x1_2 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(conv1x1_1)
    
    main_path = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(conv1x1_2)
    main_path = Dropout(rate=0.5)(main_path)

    # Branch pathway
    branch_path = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_layer)

    # Fusing paths
    fused = Add()([main_path, branch_path])

    # Final layers
    global_avg_pool = GlobalAveragePooling2D()(fused)
    flatten_layer = Flatten()(global_avg_pool)
    output_layer = Dense(units=10, activation='softmax')(flatten_layer)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model