from keras.models import Model
from keras.layers import Input, AveragePooling2D, Conv2D, Flatten, Dense, Dropout

def dl_model():
    
    input_layer = Input(shape=(28,28,1))
    
    ave_pool = AveragePooling2D(pool_size=(5, 5), strides=(3, 3), padding='valid')(input_layer)
    conv = Conv2D(filters=128, kernel_size=(1, 1), strides=(1, 1), padding='same',activation='relu')(ave_pool)
    flat = Flatten()(conv)
    dense1 = Dense(units=1024, activation='relu')(flat)
    dropout = Dropout(rate=0.3)(dense1)
    output_layer = Dense(units=1000, activation='softmax')(dropout)

    model = Model(inputs=input_layer, outputs=output_layer)

    return model
