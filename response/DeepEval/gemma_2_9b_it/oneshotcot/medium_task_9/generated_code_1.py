import keras
from keras.layers import Input, Conv2D, BatchNormalization, ReLU, Add, AveragePooling2D, Flatten, Dense

def dl_model():     
    input_layer = Input(shape=(32, 32, 3))
    
    x = Conv2D(filters=16, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_layer)

    def basic_block(input_tensor):
      x1 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_tensor)
      x1 = BatchNormalization()(x1)
      return x1

    # Block 1
    branch = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(x) 
    x = basic_block(x)
    x = Add()([x, branch])
    
    # Block 2
    branch = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(x) 
    x = basic_block(x)
    x = Add()([x, branch]) 

    x = AveragePooling2D(pool_size=(2, 2), strides=2, padding='valid')(x)
    x = Flatten()(x)
    output_layer = Dense(units=10, activation='softmax')(x)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model