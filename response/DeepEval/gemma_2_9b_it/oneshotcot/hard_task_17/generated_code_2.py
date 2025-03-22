import keras
from keras.layers import Input, Conv2D, MaxPooling2D, GlobalAveragePooling2D, Dense, Reshape, Add

def dl_model():
    input_layer = Input(shape=(32, 32, 3)) 

    # Block 1
    x = GlobalAveragePooling2D()(input_layer)
    x = Dense(128, activation='relu')(x)  
    x = Dense(32, activation='relu')(x) 
    x = Reshape((32, 32, 1))(x) # Reshape to match input shape
    weighted_features = x * input_layer 

    # Block 2
    y = Conv2D(filters=16, kernel_size=(3, 3), activation='relu')(input_layer)
    y = Conv2D(filters=32, kernel_size=(3, 3), activation='relu')(y)
    y = MaxPooling2D(pool_size=(2, 2))(y)

    # Fusion
    combined_features = Add()([weighted_features, y])

    # Classification
    x = Flatten()(combined_features)
    x = Dense(128, activation='relu')(x)
    output_layer = Dense(10, activation='softmax')(x)  

    model = keras.Model(inputs=input_layer, outputs=output_layer)
    return model