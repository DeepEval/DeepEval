import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, GlobalAveragePooling2D, Dense, Add, Multiply

def dl_model():
    # Define the input layer
    input_layer = Input(shape=(32, 32, 3))

    # Main path
    x = Conv2D(32, (3, 3), activation='relu')(input_layer)
    x = Conv2D(32, (3, 3), activation='relu')(x)
    x = GlobalAveragePooling2D()(x)
    x = Dense(128, activation='relu')(x)
    weights = Dense(32*32*3, activation='sigmoid')(x)
    weights = tf.reshape(weights, (-1, 32, 32, 3))
    main_path_output = Multiply()([input_layer, weights])

    # Branch path
    branch_path_output = input_layer

    # Combine the outputs from both paths
    combined_output = Add()([main_path_output, branch_path_output])

    # Pass through two fully connected layers
    final_output = Dense(10, activation='softmax')(combined_output)

    # Define the model
    model = Model(inputs=input_layer, outputs=final_output)

    return model

# Example usage:
# model = dl_model()
# model.summary()