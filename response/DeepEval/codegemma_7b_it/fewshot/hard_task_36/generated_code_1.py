import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Lambda, Flatten, Concatenate, Dense, Dropout, GlobalAveragePooling2D, Reshape

def block(input_tensor):
    conv1 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_tensor)
    conv2 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(conv1)
    max_pool = MaxPooling2D(pool_size=(2, 2), strides=(1, 1), padding='same')(conv2)
    dropout = Dropout(0.5)(max_pool)
    return dropout

def dl_model():
    input_layer = Input(shape=(28, 28, 1))

    main_pathway = block(input_tensor=input_layer)
    branch_pathway = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_layer)

    fused = keras.layers.add([main_pathway, branch_pathway])
    global_avg_pool = GlobalAveragePooling2D()(fused)
    flatten = Flatten()(global_avg_pool)
    output_layer = Dense(units=10, activation='softmax')(flatten)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model