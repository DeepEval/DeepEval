import keras
from keras.layers import Input, Conv2D, AveragePooling2D, Multiply, Reshape, Dense, Add, GlobalAveragePooling2D, Conv2DTranspose
from keras.models import Model
from keras.regularizers import l2

def dl_model():     

    input_layer = Input(shape=(32, 32, 3))
    
    # Main path
    conv = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_layer)
    avg_pool = GlobalAveragePooling2D()(conv)
    dense1 = Dense(units=64, activation='relu')(avg_pool)
    weights = Dense(units=32, activation='linear')(dense1)  # weights of the same size as the input layer
    weights = Reshape((1, 1, 32))(weights)
    feature_map = Multiply()([conv, weights])

    # Branch path
    conv_branch = Conv2DTranspose(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same')(conv)
    combined = Add()([feature_map, conv_branch])

    # Fully connected layers
    flatten_layer = Flatten()(combined)
    dense2 = Dense(units=64, activation='relu')(flatten_layer)
    dense3 = Dense(units=32, activation='relu')(dense2)
    output_layer = Dense(units=10, activation='softmax')(dense3)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model

model = dl_model()