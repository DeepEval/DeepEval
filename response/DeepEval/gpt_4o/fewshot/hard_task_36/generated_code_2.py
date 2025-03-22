import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Dropout, Add, GlobalAveragePooling2D, Flatten, Dense

def dl_model():
    input_layer = Input(shape=(28, 28, 1))
    
    # Main Pathway
    conv_main_1 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_layer)
    conv_main_2 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(conv_main_1)
    conv_main_3 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(conv_main_2)
    pool_main = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(conv_main_3)
    dropout_main = Dropout(0.5)(pool_main)
    
    # Branch Pathway
    conv_branch = Conv2D(filters=32, kernel_size=(3, 3), strides=(2, 2), padding='same', activation='relu')(input_layer)
    
    # Fuse both pathways
    fused_layer = Add()([dropout_main, conv_branch])
    
    # Concluding layers
    global_avg_pool = GlobalAveragePooling2D()(fused_layer)
    flatten_layer = Flatten()(global_avg_pool)
    output_layer = Dense(units=10, activation='softmax')(flatten_layer)
    
    model = keras.Model(inputs=input_layer, outputs=output_layer)
    
    return model