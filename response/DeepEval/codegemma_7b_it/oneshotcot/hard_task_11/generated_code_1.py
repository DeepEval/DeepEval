import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, BatchNormalization, Flatten, Dense, Add

def dl_model():
    input_layer = Input(shape=(32, 32, 3))
    
    main_pathway = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_layer)
    
    parallel_pathway = Concatenate()([
        Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_layer),
        Conv2D(filters=32, kernel_size=(1, 3), strides=(1, 1), padding='same', activation='relu')(input_layer),
        Conv2D(filters=32, kernel_size=(3, 1), strides=(1, 1), padding='same', activation='relu')(input_layer)
    ])
    
    concat_pathways = Concatenate()([main_pathway, parallel_pathway])
    
    main_pathway = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(concat_pathways)
    
    direct_connection = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_layer)
    
    fusion = Add()([main_pathway, direct_connection])
    
    max_pooling = MaxPooling2D(pool_size=(2, 2), strides=2, padding='valid')(fusion)
    
    bath_norm = BatchNormalization()(max_pooling)
    
    flatten_layer = Flatten()(bath_norm)
    
    dense1 = Dense(units=128, activation='relu')(flatten_layer)
    
    dense2 = Dense(units=64, activation='relu')(dense1)
    
    output_layer = Dense(units=10, activation='softmax')(dense2)
    
    model = keras.Model(inputs=input_layer, outputs=output_layer)
    
    return model

model = dl_model()