import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, GlobalAveragePooling2D, Dense, Add, Multiply

def dl_model():
    # Define the input layer
    input_layer = Input(shape=(32, 32, 3))

    # Main path
    x = Conv2D(64, (3, 3), activation='relu', padding='same')(input_layer)
    x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = GlobalAveragePooling2D()(x)
    x = Dense(64, activation='relu')(x)
    x = Dense(64, activation='relu')(x)
    weights = Dense(64, activation='sigmoid')(x)
    weights = tf.reshape(weights, (-1, 64, 1, 1))
    main_path_output = Multiply()([input_layer, weights])

    # Branch path
    branch_path_output = input_layer

    # Combine both paths
    combined_output = Add()([main_path_output, branch_path_output])

    # Additional fully connected layers
    x = Dense(128, activation='relu')(combined_output)
    x = Dense(128, activation='relu')(x)
    output_layer = Dense(10, activation='softmax')(x)

    # Define the model
    model = Model(inputs=input_layer, outputs=output_layer)

    return model

# Example usage:
# model = dl_model()
# model.summary()