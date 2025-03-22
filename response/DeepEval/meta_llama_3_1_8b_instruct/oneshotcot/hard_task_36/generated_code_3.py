import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Add, GlobalAveragePooling2D, Flatten, Dense, Dropout

def dl_model():     

    input_layer = Input(shape=(28, 28, 1))
    
    # Main Pathway
    conv = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_layer)
    conv_path1 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(conv)
    conv_path2 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(conv_path1)
    max_pooling = MaxPooling2D(pool_size=(2, 2), strides=2, padding='same')(conv_path2)
    dropout = Dropout(0.5)(max_pooling)
    
    # Branch Pathway
    conv_branch = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_layer)
    
    # Fuse the outputs from both pathways
    fuse = Add()([dropout, conv_branch])
    
    # Global Average Pooling and Flatten
    global_avg_pool = GlobalAveragePooling2D()(fuse)
    flatten_layer = Flatten()(global_avg_pool)
    
    # Output Layer
    output_layer = Dense(units=10, activation='softmax')(flatten_layer)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model