import keras
from keras import layers

def dl_model():
    
    inputs = keras.Input(shape=(32, 32, 3))
    
    # Block 1
    x = layers.Conv2D(filters=64, kernel_size=(3, 3), padding='same')(inputs)
    x = layers.Activation('relu')(x)
    x = layers.Dropout(0.2)(x)
    
    residual = layers.Conv2D(filters=64, kernel_size=(3, 3), padding='same')(x)
    residual = layers.BatchNormalization()(residual)
    
    shortcut = layers.Conv2D(filters=64, kernel_size=(1, 1), padding='same')(inputs)
    shortcut = layers.BatchNormalization()(shortcut)
    
    x = layers.Add()([x, residual, shortcut])
    x = layers.Activation('relu')(x)
    x = layers.BatchNormalization()(x)
    
    # Block 2
    x = layers.Lambda(lambda y: tf.split(y, 3, axis=-1))(x)
    
    conv1 = layers.Conv2D(filters=128, kernel_size=(1, 1), padding='same')(x[0])
    conv1 = layers.Activation('relu')(conv1)
    conv1 = layers.Dropout(0.2)(conv1)
    conv1 = layers.Conv2D(filters=128, kernel_size=(3, 3), padding='same')(conv1)
    conv1 = layers.Activation('relu')(conv1)
    conv1 = layers.Dropout(0.2)(conv1)
    conv1 = layers.Conv2D(filters=64, kernel_size=(1, 1), padding='same')(conv1)
    conv1 = layers.Activation('relu')(conv1)
    conv1 = layers.BatchNormalization()(conv1)
    
    conv2 = layers.Conv2D(filters=128, kernel_size=(1, 1), padding='same')(x[1])
    conv2 = layers.Activation('relu')(conv2)
    conv2 = layers.Dropout(0.2)(conv2)
    conv2 = layers.Conv2D(filters=128, kernel_size=(5, 5), padding='same')(conv2)
    conv2 = layers.Activation('relu')(conv2)
    conv2 = layers.Dropout(0.2)(conv2)
    conv2 = layers.Conv2D(filters=64, kernel_size=(1, 1), padding='same')(conv2)
    conv2 = layers.Activation('relu')(conv2)
    conv2 = layers.BatchNormalization()(conv2)
    
    conv3 = layers.Conv2D(filters=128, kernel_size=(1, 1), padding='same')(x[2])
    conv3 = layers.Activation('relu')(conv3)
    conv3 = layers.Dropout(0.2)(conv3)
    conv3 = layers.Conv2D(filters=128, kernel_size=(7, 7), padding='same')(conv3)
    conv3 = layers.Activation('relu')(conv3)
    conv3 = layers.Dropout(0.2)(conv3)
    conv3 = layers.Conv2D(filters=64, kernel_size=(1, 1), padding='same')(conv3)
    conv3 = layers.Activation('relu')(conv3)
    conv3 = layers.BatchNormalization()(conv3)
    
    x = layers.Concatenate()([conv1, conv2, conv3])
    
    # Output
    x = layers.Flatten()(x)
    outputs = layers.Dense(units=10, activation='softmax')(x)
    
    model = keras.Model(inputs=inputs, outputs=outputs)
    return model