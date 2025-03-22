import keras
from keras.layers import Input, Conv2D, GlobalAveragePooling2D, Dense, Multiply, Add, Flatten
from keras.models import Model

def dl_model():
    input_layer = Input(shape=(32, 32, 3))
    
    # Main path
    global_avg_pool = GlobalAveragePooling2D()(input_layer)
    dense1_main = Dense(units=32, activation='relu')(global_avg_pool)
    dense2_main = Dense(units=3, activation='sigmoid')(dense1_main)  # Assuming 3 channels in input

    # Reshape and multiply weights with the original input
    reshape_weights = keras.layers.Reshape((1, 1, 3))(dense2_main)
    scaled_input = Multiply()([input_layer, reshape_weights])

    # Branch path
    branch_conv = Conv2D(filters=3, kernel_size=(3, 3), padding='same', activation='relu')(input_layer)

    # Combine paths
    combined = Add()([scaled_input, branch_conv])

    # Fully connected layers for classification
    flatten = Flatten()(combined)
    dense1 = Dense(units=128, activation='relu')(flatten)
    dense2 = Dense(units=64, activation='relu')(dense1)
    output_layer = Dense(units=10, activation='softmax')(dense2)  # CIFAR-10 has 10 classes

    model = Model(inputs=input_layer, outputs=output_layer)

    return model