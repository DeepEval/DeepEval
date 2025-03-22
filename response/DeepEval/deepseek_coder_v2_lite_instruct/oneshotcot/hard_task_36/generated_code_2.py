import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Dropout, Add, GlobalAveragePooling2D, Flatten, Dense
from keras.models import Model

def dl_model():
    input_layer = Input(shape=(28, 28, 1))

    # Main pathway
    conv_main = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_layer)
    conv_1x1_1 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(conv_main)
    conv_1x1_2 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(conv_1x1_1)
    max_pooling = MaxPooling2D(pool_size=(2, 2), strides=2, padding='same')(conv_1x1_2)
    dropout = Dropout(0.5)(max_pooling)

    # Branch pathway
    conv_branch = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_layer)

    # Fusion of both pathways
    fused = Add()([dropout, conv_branch])

    # Global Average Pooling
    global_avg_pool = GlobalAveragePooling2D()(fused)

    # Flatten
    flatten_layer = Flatten()(global_avg_pool)

    # Fully connected layer
    dense_layer = Dense(units=128, activation='relu')(flatten_layer)
    output_layer = Dense(units=10, activation='softmax')(dense_layer)

    model = Model(inputs=input_layer, outputs=output_layer)

    return model