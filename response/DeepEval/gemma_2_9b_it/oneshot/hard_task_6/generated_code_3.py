import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, BatchNormalization, Flatten, Dense, Lambda, Permute

def dl_model():
    
    input_layer = Input(shape=(32, 32, 3))
    
    # Branch Path
    branch_pool = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(input_layer)

    # Main Path
    x = input_layer 

    # Block 1
    x = Lambda(lambda x: tf.split(x, num_or_size_splits=3, axis=3))(x)
    x = [Conv2D(filters=int(x.shape[3]/3), kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(i) for i in x]
    x = Concatenate()([i for i in x])

    # Block 2
    x = Lambda(lambda x: tf.keras.backend.shape(x)[1:3])(x)
    x = tf.reshape(x, (-1,  tf.shape(x)[1], tf.shape(x)[2], 3, int(tf.shape(x)[3]/3)))
    x = Permute((2, 3, 1, 4, 5))(x)
    x = tf.reshape(x, (-1,  tf.shape(x)[1], tf.shape(x)[2], tf.shape(x)[3]*3))

    # Block 3
    x = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu', depth_wise=True)(x)

    # Concatenation
    x = Concatenate()([x, branch_pool])

    # Flatten and Fully Connected Layers
    x = Flatten()(x)
    x = Dense(units=128, activation='relu')(x)
    output_layer = Dense(units=10, activation='softmax')(x)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model