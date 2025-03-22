import keras
from keras.layers import Input, Conv2D, Dropout, MaxPooling2D, Concatenate, Flatten, Dense

def dl_model():     
    
    input_layer = Input(shape=(28, 28, 1))

    # Main Path
    x = Conv2D(filters=32, kernel_size=(3, 3), activation='relu', padding='same')(input_layer)
    x = Dropout(0.25)(x)
    x = Conv2D(filters=64, kernel_size=(3, 3), activation='relu', padding='same')(x)
    x = Conv2D(filters=32, kernel_size=(3, 3), activation='relu', padding='same')(x) 

    # Branch Path
    branch_x = Conv2D(filters=64, kernel_size=(3, 3), activation='relu', padding='same')(input_layer)

    # Combine Paths
    x = Concatenate()([x, branch_x])

    # Flatten and Classify
    x = Flatten()(x)
    output_layer = Dense(units=10, activation='softmax')(x)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model