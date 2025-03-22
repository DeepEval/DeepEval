import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, Lambda, BatchNormalization, Flatten, Dense, Permute
from keras.datasets import cifar10
from keras.preprocessing.image import ImageDataGenerator

def dl_model():     

    input_layer = Input(shape=(32, 32, 3))

    # Main Path
    
    # Block 1
    x = Lambda(lambda x: tf.split(x, num_or_size_splits=3, axis=3))(input_layer)
    x = [Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(xi) for xi in x]
    x = Concatenate()([xi for xi in x])
    
    # Block 2
    x = Lambda(lambda x: tf.keras.backend.shape(x) )
    x = Lambda(lambda x: tf.reshape(x, [-1, 32, 32, 3, 1]))(x)
    x = Permute((1, 2, 4, 3))(x)
    x = Lambda(lambda x: tf.reshape(x, [-1, 32, 32, 3]))(x)

    # Block 3
    x = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu', use_bias=False)(x)
    x = BatchNormalization()(x) 

    # Branch Path
    branch_output = Lambda(lambda x: tf.reduce_mean(x, axis=(1,2)))(input_layer)

    # Concatenation
    x = Concatenate()([x, branch_output])

    # Flatten and fully connected layers
    x = Flatten()(x)
    x = Dense(units=128, activation='relu')(x)
    output_layer = Dense(units=10, activation='softmax')(x)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model