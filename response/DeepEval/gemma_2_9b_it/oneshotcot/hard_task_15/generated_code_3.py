import keras
from keras.layers import Input, GlobalAveragePooling2D, Dense, Reshape, Conv2D, Add

def dl_model():

    input_layer = Input(shape=(32, 32, 3))

    # Main Path
    x = GlobalAveragePooling2D()(input_layer)
    x = Dense(units=32, activation='relu')(x)
    x = Dense(units=32, activation='relu')(x) 
    x = Reshape((32, 32, 3))(x) # Reshape to match input layer

    # Branch Path
    branch_path = input_layer

    # Combine paths
    combined = Add()([x, branch_path])

    # Final Classification Layers
    output_layer = Dense(units=10, activation='softmax')(combined)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model