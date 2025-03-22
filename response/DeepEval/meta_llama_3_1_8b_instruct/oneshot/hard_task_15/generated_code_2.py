import keras
from keras.layers import Input, GlobalAveragePooling2D, Dense, Reshape, Multiply, Add
from keras.models import Model

def dl_model():     

    input_layer = Input(shape=(32, 32, 3))
    
    # Main Path
    conv1 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_layer)
    max_pooling1 = MaxPooling2D(pool_size=(3, 3), strides=2, padding='same')(conv1)
    conv2 = Conv2D(filters=128, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(max_pooling1)
    max_pooling2 = MaxPooling2D(pool_size=(3, 3), strides=2, padding='same')(conv2)
    global_avg_pooling = GlobalAveragePooling2D()(max_pooling2)
    dense1 = Dense(units=128, activation='relu')(global_avg_pooling)
    weights = Dense(units=64, activation='linear')(dense1)
    weights = Reshape((3, 3, 1))(weights)
    weighted_input = Multiply()([conv2, weights])
    
    # Branch Path
    branch_conv = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_layer)
    
    # Combine Main and Branch Paths
    combined_output = Add()([weighted_input, branch_conv])
    
    # Final Fully Connected Layers
    flat = Flatten()(combined_output)
    dense2 = Dense(units=128, activation='relu')(flat)
    output_layer = Dense(units=10, activation='softmax')(dense2)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model