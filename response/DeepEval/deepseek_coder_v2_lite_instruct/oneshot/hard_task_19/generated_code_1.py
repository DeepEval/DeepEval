import keras
from keras.layers import Input, Conv2D, MaxPooling2D, GlobalAveragePooling2D, Dense, Multiply, Add, BatchNormalization, Flatten
from keras.models import Model

def dl_model():
    input_layer = Input(shape=(32, 32, 3))
    
    # Main path
    main_path = Conv2D(filters=32, kernel_size=(3, 3), padding='same', activation='relu')(input_layer)
    main_path = Conv2D(filters=32, kernel_size=(3, 3), padding='same', activation='relu')(main_path)
    main_path = Conv2D(filters=32, kernel_size=(3, 3), padding='same', activation='relu')(main_path)
    main_path = MaxPooling2D(pool_size=(2, 2))(main_path)
    
    # Branch path
    branch_path = GlobalAveragePooling2D()(input_layer)
    branch_path = Dense(units=128, activation='relu')(branch_path)
    branch_path = Dense(units=64, activation='relu')(branch_path)
    branch_weights = Dense(units=32, activation='relu')(branch_path)
    branch_weights = Dense(units=3, activation='softmax')(branch_weights)
    branch_weights = keras.backend.reshape(branch_weights, (-1, 32, 32, 3))
    weighted_input = Multiply()([input_layer, branch_weights])
    
    # Combine outputs from both paths
    combined = Add()([main_path, weighted_input])
    
    # Additional fully connected layers
    combined = Flatten()(combined)
    combined = Dense(units=128, activation='relu')(combined)
    output_layer = Dense(units=10, activation='softmax')(combined)
    
    model = Model(inputs=input_layer, outputs=output_layer)
    
    return model