from tensorflow import keras
from tensorflow.keras import layers

def dl_model():
    input_tensor = keras.Input(shape=(32, 32, 3))

    # Increase channel dimensionality threefold
    x = layers.Conv2D(filters=3 * 3, kernel_size=(1, 1), activation='relu')(input_tensor)

    # Extract initial features with depthwise separable convolution
    x = layers.DepthwiseConv2D(kernel_size=(3, 3), activation='relu')(x)

    # Channel attention module
    
    # Global average pooling
    global_pool = layers.GlobalAveragePooling2D()(x)

    # Two fully connected layers to generate channel attention weights
    attention_weights = layers.Dense(units=int(x.shape[-1]), activation='sigmoid')(global_pool) 

    # Reshape attention weights to match the initial features
    attention_weights = layers.Reshape((x.shape[1], x.shape[2], 1))(attention_weights) 

    # Multiply attention weights with initial features
    x = layers.multiply([x, attention_weights])

    # Reduce dimensionality
    x = layers.Conv2D(filters=3, kernel_size=(1, 1), activation='relu')(x)

    # Add initial input to enhance features
    x = layers.Add()([input_tensor, x])

    # Flatten and classify
    x = layers.Flatten()(x)
    output_tensor = layers.Dense(units=10, activation='softmax')(x)

    # Create the model
    model = keras.Model(inputs=input_tensor, outputs=output_tensor)
    return model

# Get the model
model = dl_model()

# Print model summary
model.summary()