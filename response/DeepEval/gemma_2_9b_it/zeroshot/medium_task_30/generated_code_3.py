import tensorflow as tf
from tensorflow.keras import layers

def dl_model():
    input_tensor = layers.Input(shape=(32, 32, 3))

    # First average pooling layer with 1x1 pooling window and stride
    x = layers.AveragePooling2D(pool_size=(1, 1), strides=(1, 1))(input_tensor) 

    # Second average pooling layer with 2x2 pooling window and stride
    x = layers.AveragePooling2D(pool_size=(2, 2), strides=(2, 2))(x)

    # Third average pooling layer with 4x4 pooling window and stride
    x = layers.AveragePooling2D(pool_size=(4, 4), strides=(4, 4))(x)

    # Flatten the outputs of the pooling layers
    x = layers.Flatten()(x)

    # Concatenate the flattened features
    x = layers.Concatenate()([x, x, x])  

    # Flatten again
    x = layers.Flatten()(x)

    # Fully connected layers
    x = layers.Dense(128, activation='relu')(x)
    outputs = layers.Dense(10, activation='softmax')(x)

    model = tf.keras.Model(inputs=input_tensor, outputs=outputs)
    return model

# Example usage
model = dl_model()
model.summary()