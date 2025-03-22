import keras
from keras.layers import Input, SeparableConv2D, MaxPooling2D, Conv2D, Add, BatchNormalization, Flatten, Dense
from keras.layers import GlobalAveragePooling2D
from tensorflow.keras import layers

def dl_model():
    input_layer = Input(shape=(28, 28, 1))
    
    # Main path
    def main_path(input_tensor):
        block1 = layers.Layer()
        relu1 = layers.Activation('relu')(input_tensor)
        block1 = relu1
        separable_conv1 = SeparableConv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(block1)
        max_pooling1 = MaxPooling2D(pool_size=(2, 2), strides=2, padding='same')(separable_conv1)

        block2 = layers.Layer()
        relu2 = layers.Activation('relu')(max_pooling1)
        block2 = relu2
        separable_conv2 = SeparableConv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(block2)
        max_pooling2 = MaxPooling2D(pool_size=(2, 2), strides=2, padding='same')(separable_conv2)

        return max_pooling2
    
    main_output = main_path(input_layer)

    # Branch path
    branch = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_layer)
    
    # Merge main and branch paths
    merged = Add()([main_output, branch])
    bath_norm = BatchNormalization()(merged)
    global_avg_pool = GlobalAveragePooling2D()(bath_norm)
    flatten_layer = Flatten()(global_avg_pool)
    dense = Dense(units=10, activation='softmax')(flatten_layer)
    
    model = keras.Model(inputs=input_layer, outputs=dense)

    return model