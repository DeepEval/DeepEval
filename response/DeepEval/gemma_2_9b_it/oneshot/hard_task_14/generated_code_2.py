import keras
from keras.layers import Input, Conv2D, GlobalAveragePooling2D, Dense, Reshape, Add

def dl_model():
    
    input_layer = Input(shape=(32, 32, 3))

    # Main Path
    x = GlobalAveragePooling2D()(input_layer)
    x = Dense(units=32, activation='relu')(x)  
    x = Dense(units=32, activation='relu')(x)  
    x = Dense(units=3 * 32, activation='linear')(x)  
    x = Reshape((3, 32))(x)
    
    # Branch Path
    branch_x = Conv2D(filters=3 * 32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_layer) 

    # Add outputs from both paths
    output = Add()([x, branch_x])

    # Final fully connected layers
    output = Flatten()(output)
    output = Dense(units=128, activation='relu')(output)
    output = Dense(units=64, activation='relu')(output)
    output_layer = Dense(units=10, activation='softmax')(output)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model