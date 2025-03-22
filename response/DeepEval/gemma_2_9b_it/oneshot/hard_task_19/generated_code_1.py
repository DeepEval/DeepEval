import keras
from keras.layers import Input, Conv2D, MaxPooling2D, GlobalAveragePooling2D, Flatten, Dense, Reshape, Multiply

def dl_model():     

    input_layer = Input(shape=(32, 32, 3))
    
    # Main Path
    conv1 = Conv2D(filters=16, kernel_size=(3, 3), activation='relu')(input_layer)
    conv2 = Conv2D(filters=32, kernel_size=(3, 3), activation='relu')(conv1)
    conv3 = Conv2D(filters=64, kernel_size=(3, 3), activation='relu')(conv2)
    max_pooling = MaxPooling2D(pool_size=(2, 2))(conv3)

    # Branch Path
    avg_pooling = GlobalAveragePooling2D()(input_layer)
    dense4 = Dense(units=128, activation='relu')(avg_pooling)
    dense5 = Dense(units=64, activation='relu')(dense4)
    channel_weights = Reshape((32, 32, 64))(dense5)  

    # Multiply channel weights with input
    weighted_input = Multiply()([input_layer, channel_weights])

    # Concatenate outputs
    concat_layer = keras.layers.Concatenate()([max_pooling, weighted_input])
    
    flatten_layer = Flatten()(concat_layer)
    dense6 = Dense(units=128, activation='relu')(flatten_layer)
    output_layer = Dense(units=10, activation='softmax')(dense6)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model