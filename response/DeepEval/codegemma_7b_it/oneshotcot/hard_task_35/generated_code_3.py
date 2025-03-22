import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, BatchNormalization, Flatten, Dense, GlobalAveragePooling2D, Reshape

def dl_model():
    
    input_layer = Input(shape=(32, 32, 3))
    
    branch_1 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='valid', activation='relu')(input_layer)
    branch_1 = MaxPooling2D(pool_size=(2, 2), strides=2, padding='valid')(branch_1)
    branch_1 = GlobalAveragePooling2D()(branch_1)
    branch_1 = Dense(units=256, activation='relu')(branch_1)
    branch_1 = Dense(units=32, activation='relu')(branch_1)
    branch_1 = Reshape(target_shape=(32, 1, 1))(branch_1)
    
    branch_2 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='valid', activation='relu')(input_layer)
    branch_2 = MaxPooling2D(pool_size=(2, 2), strides=2, padding='valid')(branch_2)
    branch_2 = GlobalAveragePooling2D()(branch_2)
    branch_2 = Dense(units=256, activation='relu')(branch_2)
    branch_2 = Dense(units=32, activation='relu')(branch_2)
    branch_2 = Reshape(target_shape=(32, 1, 1))(branch_2)
    
    concat = Concatenate()([branch_1, branch_2])
    flatten_layer = Flatten()(concat)
    output_layer = Dense(units=10, activation='softmax')(flatten_layer)

    model = keras.Model(inputs=input_layer, outputs=output_layer)
    
    return model