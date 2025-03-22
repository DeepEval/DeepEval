import keras
from keras.layers import Input, Conv2D, MaxPooling2D, GlobalAveragePooling2D, Dense, Add, Multiply, Flatten
from keras.models import Model

def dl_model():
    input_layer = Input(shape=(32, 32, 3))
    
    # Main path
    conv1 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_layer)
    conv2 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(conv1)
    conv3 = Conv2D(filters=128, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(conv2)
    main_path = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(conv3)
    
    # Branch path
    branch_avg_pool = GlobalAveragePooling2D()(input_layer)
    dense1 = Dense(units=64, activation='relu')(branch_avg_pool)
    dense2 = Dense(units=32, activation='relu')(dense1)
    weights = Dense(units=1, activation='sigmoid')(dense2)
    reshaped_weights = keras.layers.Reshape((1, 1, 1))(weights)
    
    # Multiply weights with input
    weighted_input = Multiply()([reshaped_weights, input_layer])
    
    # Add main path and weighted input
    added = Add()([main_path, weighted_input])
    
    # Additional fully connected layers
    flatten = Flatten()(added)
    dense3 = Dense(units=128, activation='relu')(flatten)
    output_layer = Dense(units=10, activation='softmax')(dense3)
    
    model = Model(inputs=input_layer, outputs=output_layer)
    
    return model