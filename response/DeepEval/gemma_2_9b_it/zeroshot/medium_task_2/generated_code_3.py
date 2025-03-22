from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, concatenate

def dl_model():
    # Input Layer
    input_tensor = Input(shape=(32, 32, 3)) 

    # Main Path
    x_main = Conv2D(32, (3, 3), activation='relu')(input_tensor)
    x_main = Conv2D(64, (3, 3), activation='relu')(x_main)
    x_main = MaxPooling2D((2, 2))(x_main)

    # Branch Path
    x_branch = Conv2D(16, (5, 5), activation='relu')(input_tensor)

    # Concatenate Features
    x_combined = concatenate([x_main, x_branch], axis=-1)

    # Flatten and Fully Connected Layers
    x_flat = Flatten()(x_combined)
    x_dense1 = Dense(128, activation='relu')(x_flat)
    output_layer = Dense(10, activation='softmax')(x_dense1)

    # Construct Model
    model = Model(inputs=input_tensor, outputs=output_layer)
    
    return model