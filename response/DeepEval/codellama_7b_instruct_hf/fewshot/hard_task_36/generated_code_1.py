import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Dropout, Flatten, Dense

def dl_model():
    input_layer = Input(shape=(28, 28, 1))

    # Main pathway
    conv1 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_layer)
    conv2 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(conv1)
    max_pooling = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(conv2)
    dropout = Dropout(rate=0.5)(max_pooling)

    # Branch pathway
    branch_conv = Conv2D(filters=32, kernel_size=(5, 5), strides=(1, 1), padding='same', activation='relu')(input_layer)

    # Fuse outputs
    fused_output = Add()([max_pooling, branch_conv])

    # Global average pooling
    global_avg_pooling = GlobalAveragePooling2D()(fused_output)

    # Flatten
    flatten = Flatten()(global_avg_pooling)

    # Fully connected layer
    output_layer = Dense(units=10, activation='softmax')(flatten)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model