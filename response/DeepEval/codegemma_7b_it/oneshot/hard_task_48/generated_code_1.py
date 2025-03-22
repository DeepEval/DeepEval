import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Lambda, BatchNormalization, Flatten, Dense, Dropout, concatenate

def dl_model():
    
    input_layer = Input(shape=(32, 32, 3))

    # Block 1
    def block1(input_tensor):
        x1 = Lambda(lambda x: tf.split(x, num_or_size_splits=3, axis=3))(input_tensor)
        x2 = Lambda(lambda x: tf.split(x, num_or_size_splits=3, axis=3))(input_tensor)
        x3 = Lambda(lambda x: tf.split(x, num_or_size_splits=3, axis=3))(input_tensor)
        
        x1 = Lambda(lambda x: Conv2D(filters=64, kernel_size=(1, 1), padding='same', activation='relu')(x))(x1)
        x2 = Lambda(lambda x: Conv2D(filters=64, kernel_size=(3, 3), padding='same', activation='relu')(x))(x2)
        x3 = Lambda(lambda x: Conv2D(filters=64, kernel_size=(5, 5), padding='same', activation='relu')(x))(x3)
        
        x1 = Lambda(lambda x: BatchNormalization()(x))(x1)
        x2 = Lambda(lambda x: BatchNormalization()(x))(x2)
        x3 = Lambda(lambda x: BatchNormalization()(x))(x3)
        
        return concatenate([x1, x2, x3])
    
    # Block 2
    def block2(input_tensor):
        p1 = Lambda(lambda x: Conv2D(filters=64, kernel_size=(1, 1), padding='same', activation='relu')(x))(input_tensor)
        p2 = Lambda(lambda x: MaxPooling2D(pool_size=(3, 3), padding='same')(x))(input_tensor)
        p2 = Lambda(lambda x: Conv2D(filters=64, kernel_size=(1, 1), padding='same', activation='relu')(p2))(p2)
        p3 = Lambda(lambda x: Conv2D(filters=64, kernel_size=(1, 1), padding='same', activation='relu')(x))(input_tensor)
        p3 = Lambda(lambda x: Conv2D(filters=64, kernel_size=(1, 3), padding='same', activation='relu')(p3))(p3)
        p3 = Lambda(lambda x: Conv2D(filters=64, kernel_size=(3, 1), padding='same', activation='relu')(p3))(p3)
        p4 = Lambda(lambda x: Conv2D(filters=64, kernel_size=(1, 1), padding='same', activation='relu')(x))(input_tensor)
        p4 = Lambda(lambda x: Conv2D(filters=64, kernel_size=(3, 3), padding='same', activation='relu')(p4))(p4)
        
        p1 = Lambda(lambda x: BatchNormalization()(x))(p1)
        p2 = Lambda(lambda x: BatchNormalization()(x))(p2)
        p3 = Lambda(lambda x: BatchNormalization()(x))(p3)
        p4 = Lambda(lambda x: BatchNormalization()(x))(p4)
        
        return concatenate([p1, p2, p3, p4])
    
    # Model Construction
    x = block1(input_layer)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = block2(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = Flatten()(x)
    x = Dense(units=512, activation='relu')(x)
    x = Dropout(rate=0.5)(x)
    x = Dense(units=10, activation='softmax')(x)

    model = keras.Model(inputs=input_layer, outputs=x)
    
    return model