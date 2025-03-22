import keras
from keras.layers import Input, Conv2D, GlobalAveragePooling2D, Concatenate, Dense, Reshape

def dl_model(): 
    input_layer = Input(shape=(32, 32, 3))

    # Main Path
    x = GlobalAveragePooling2D()(input_layer)
    x = Dense(units=32, activation='relu')(x)
    x = Dense(units=3, activation='relu')(x)
    x = Reshape((32, 32, 3))(x)  

    # Branch Path
    branch_layer = Conv2D(filters=3, kernel_size=(3, 3), activation='relu')(input_layer)

    # Combine Paths
    combined_tensor = x + branch_layer 

    # Final Classification Layers
    x = Flatten()(combined_tensor)
    x = Dense(units=128, activation='relu')(x)
    x = Dense(units=64, activation='relu')(x)
    output_layer = Dense(units=10, activation='softmax')(x)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model