import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, GlobalAveragePooling2D, Dense, Multiply, Add

def dl_model():
    # Define the input layer
    input_layer = Input(shape=(32, 32, 3))

    # Main path
    x = Conv2D(64, (3, 3), activation='relu')(input_layer)
    x = Conv2D(64, (3, 3), activation='relu')(x)
    x = GlobalAveragePooling2D()(x)
    x = Dense(64, activation='relu')(x)
    x = Dense(64)(x)
    weights = tf.reshape(x, (-1, 32, 32, 3))
    x = Multiply()([input_layer, weights])

    # Branch path
    y = input_layer

    # Combine the outputs from both paths
    combined = Add()([x, y])

    # Final fully connected layers
    z = GlobalAveragePooling2D()(combined)
    output_layer = Dense(10, activation='softmax')(z)

    # Define the model
    model = Model(inputs=input_layer, outputs=output_layer)

    return model

# Example usage
model = dl_model()
model.summary()