import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, BatchNormalization, Flatten, Dense, Dropout
from keras import regularizers

def dl_model(): 
    
    input_layer = Input(shape=(28, 28, 1))
    
    # Block 1
    path1 = MaxPooling2D(pool_size=(1, 1), strides=1, padding='valid')(input_layer)
    path2 = MaxPooling2D(pool_size=(2, 2), strides=2, padding='valid')(input_layer)
    path3 = MaxPooling2D(pool_size=(4, 4), strides=4, padding='valid')(input_layer)
    
    path1 = Flatten()(path1)
    path2 = Flatten()(path2)
    path3 = Flatten()(path3)
    
    path1 = Dropout(0.2)(path1, training=True)
    path2 = Dropout(0.2)(path2, training=True)
    path3 = Dropout(0.2)(path3, training=True)
    
    concat_block1 = Concatenate()([path1, path2, path3])
    
    # Block 2
    path4 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu', kernel_regularizer=regularizers.l2(0.001))(concat_block1)
    path5 = Conv2D(filters=32, kernel_size=(1, 7), strides=(1, 1), padding='same', activation='relu', kernel_regularizer=regularizers.l2(0.001))(concat_block1)
    path5 = Conv2D(filters=32, kernel_size=(7, 1), strides=(1, 1), padding='same', activation='relu', kernel_regularizer=regularizers.l2(0.001))(path5)
    
    path6 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu', kernel_regularizer=regularizers.l2(0.001))(concat_block1)
    path6 = Conv2D(filters=32, kernel_size=(7, 1), strides=(1, 1), padding='same', activation='relu', kernel_regularizer=regularizers.l2(0.001))(path6)
    path6 = Conv2D(filters=32, kernel_size=(1, 7), strides=(1, 1), padding='same', activation='relu', kernel_regularizer=regularizers.l2(0.001))(path6)
    
    path7 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu', kernel_regularizer=regularizers.l2(0.001))(concat_block1)
    path7 = Conv2D(filters=32, kernel_size=(5, 5), strides=(1, 1), padding='same', activation='relu', kernel_regularizer=regularizers.l2(0.001))(path7)
    
    path8 = MaxPooling2D(pool_size=(1, 1), strides=1, padding='valid')(concat_block1)
    path8 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu', kernel_regularizer=regularizers.l2(0.001))(path8)
    
    path4 = Flatten()(path4)
    path5 = Flatten()(path5)
    path6 = Flatten()(path6)
    path7 = Flatten()(path7)
    path8 = Flatten()(path8)
    
    concat_block2 = Concatenate()([path4, path5, path6, path7, path8])
    concat_block2 = Dropout(0.2)(concat_block2, training=True)
    
    # Output layer
    output_layer = Dense(units=10, activation='softmax')(concat_block2)
    
    model = keras.Model(inputs=input_layer, outputs=output_layer)
    
    return model