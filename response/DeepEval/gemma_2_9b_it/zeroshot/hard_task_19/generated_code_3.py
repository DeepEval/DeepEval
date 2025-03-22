import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, GlobalAveragePooling2D, Dense, Reshape, Multiply

def dl_model():
    
    # Input Layer
    inputs = Input(shape=(32, 32, 3))

    # Main Path
    x = Conv2D(32, (3, 3), activation='relu')(inputs)
    x = Conv2D(64, (3, 3), activation='relu')(x)
    x = Conv2D(128, (3, 3), activation='relu')(x)
    x = MaxPooling2D((2, 2))(x)

    # Branch Path
    branch_inputs = inputs
    branch_x = GlobalAveragePooling2D()(branch_inputs)
    branch_x = Dense(128, activation='relu')(branch_x)
    branch_x = Dense(32 * 32 * 128, activation='relu')(branch_x)  # Reshape for channel weights
    branch_x = Reshape((32, 32, 128))(branch_x) 
    branch_x = Multiply()([branch_x, inputs]) # Multiply channel weights with input

    # Concatenate Outputs
    merged = tf.keras.layers.Concatenate()([x, branch_x])

    # Final Classification Layers
    merged = Dense(256, activation='relu')(merged)
    outputs = Dense(10, activation='softmax')(merged)

    # Create Model
    model = Model(inputs=inputs, outputs=outputs)

    return model