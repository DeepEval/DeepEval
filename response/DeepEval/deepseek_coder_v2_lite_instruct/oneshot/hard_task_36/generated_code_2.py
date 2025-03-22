import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Add, BatchNormalization, Dropout, Flatten, Dense, GlobalAveragePooling2D
from keras.models import Model

def dl_model():
    input_layer = Input(shape=(28, 28, 1))

    # Main pathway
    conv_3x3 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_layer)
    conv_1x1_1 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(conv_3x3)
    conv_1x1_2 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(conv_3x3)
    max_pooling = MaxPooling2D(pool_size=(2, 2), strides=2, padding='valid')(conv_1x1_2)
    dropout = Dropout(0.5)(max_pooling)

    # Branch pathway
    branch_conv = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(dropout)

    # Fuse outputs from both pathways
    fused = Add()([conv_1x1_1, branch_conv])

    # Global average pooling
    global_avg_pool = GlobalAveragePooling2D()(fused)

    # Flatten
    flatten_layer = Flatten()(global_avg_pool)

    # Fully connected layer
    dense1 = Dense(units=128, activation='relu')(flatten_layer)
    dense2 = Dense(units=64, activation='relu')(dense1)
    output_layer = Dense(units=10, activation='softmax')(dense2)

    model = Model(inputs=input_layer, outputs=output_layer)

    return model