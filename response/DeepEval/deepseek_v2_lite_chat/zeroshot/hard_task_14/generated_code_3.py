import tensorflow as tf
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Flatten, Conv2D, Add
from tensorflow.keras.models import Model

def dl_model():
    # Input layer
    inputs = tf.keras.Input(shape=(32, 32, 3))
    
    # Main path
    x = Conv2D(64, 3, activation='relu')(inputs)
    x = GlobalAveragePooling2D()(x)
    x = Dense(512, activation='relu')(x)
    weights1 = Dense(shape=inputs.shape[-1])(x)
    weights1 = tf.reshape(weights1, (weights1.shape[0], weights1.shape[1], 1, 1))
    x = tf.keras.layers.multiply([x, weights1])
    
    # Branch path
    x = Conv2D(64, 3, activation='relu')(inputs)
    x = GlobalAveragePooling2D()(x)
    weights2 = Dense(shape=inputs.shape[-1])(x)
    weights2 = tf.reshape(weights2, (weights2.shape[0], weights2.shape[1], 1, 1))
    x = tf.keras.layers.multiply([x, weights2])
    
    # Combine paths
    combined = tf.keras.layers.Add()([x, inputs])
    
    # Fully connected layers
    x = Dense(512, activation='relu')(combined)
    output = Dense(10, activation='softmax')(x)  # Assuming 10 classes
    
    # Model
    model = Model(inputs=[inputs], outputs=[output])
    
    return model

# To compile and print model summary
model = dl_model()
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()