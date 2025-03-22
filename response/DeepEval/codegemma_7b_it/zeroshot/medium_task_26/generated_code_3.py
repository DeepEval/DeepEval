from tensorflow.keras import Sequential, Model, Input
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, concatenate

def dl_model():
    inputs = Input(shape=(32, 32, 64))
    x = Conv2D(32, (1, 1), padding='same', activation='relu')(inputs)
    
    pathway_one = Conv2D(32, (1, 1), padding='same', activation='relu')(x)
    pathway_one = Conv2D(64, (3, 3), padding='same', activation='relu')(pathway_one)
    pathway_two = Conv2D(32, (1, 1), padding='same', activation='relu')(x)
    pathway_two = Conv2D(64, (5, 5), padding='same', activation='relu')(pathway_two)
    
    concat_layer = concatenate([pathway_one, pathway_two], axis=3)
    
    concat_layer = MaxPooling2D((2, 2), padding='same')(concat_layer)
    concat_layer = Dropout(0.25)(concat_layer)
    
    concat_layer = Flatten()(concat_layer)
    concat_layer = Dense(512, activation='relu')(concat_layer)
    concat_layer = Dropout(0.5)(concat_layer)
    
    outputs = Dense(2, activation='softmax')(concat_layer)
    
    model = Model(inputs=inputs, outputs=outputs)
    
    return model