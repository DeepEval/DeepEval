import keras
from keras.layers import Input, Conv2D, Add, Concatenate, BatchNormalization, Flatten, Dense

def dl_model():
    
    input_layer = Input(shape=(32, 32, 64))
    
    # Main Path
    dimensionality_reduction = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_layer)
    feature_extraction_1x1 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(dimensionality_reduction)
    feature_extraction_3x3 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(dimensionality_reduction)
    main_path_output = Concatenate()([feature_extraction_1x1, feature_extraction_3x3])
    
    # Branch Path
    branch_path_output = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_layer)
    
    # Combine Main and Branch Paths
    combined_output = Add()([main_path_output, branch_path_output])
    
    bath_norm = BatchNormalization()(combined_output)
    flatten_layer = Flatten()(bath_norm)
    dense1 = Dense(units=128, activation='relu')(flatten_layer)
    output_layer = Dense(units=10, activation='softmax')(dense1)
    
    model = keras.Model(inputs=input_layer, outputs=output_layer)
    
    return model