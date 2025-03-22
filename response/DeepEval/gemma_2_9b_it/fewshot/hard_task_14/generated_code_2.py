import keras
from keras.layers import Input, GlobalAveragePooling2D, Dense, Reshape, Conv2D, Add

def dl_model():
    input_layer = Input(shape=(32, 32, 3))
    
    # Main Path
    main_path = GlobalAveragePooling2D()(input_layer)
    main_path = Dense(units=32, activation='relu')(main_path)
    main_path = Dense(units=32, activation='relu')(main_path)
    main_path = Reshape((32, 32, 3))(main_path)  
    
    # Branch Path
    branch_path = Conv2D(filters=3, kernel_size=(3, 3), strides=1, padding='same', activation='relu')(input_layer)
    
    # Add outputs
    combined_output = Add()([main_path, branch_path])

    # Classification Layers
    flatten_layer = Flatten()(combined_output)
    dense1 = Dense(units=128, activation='relu')(flatten_layer)
    dense2 = Dense(units=64, activation='relu')(dense1)
    output_layer = Dense(units=10, activation='softmax')(dense2)

    model = keras.Model(inputs=input_layer, outputs=output_layer)
    return model