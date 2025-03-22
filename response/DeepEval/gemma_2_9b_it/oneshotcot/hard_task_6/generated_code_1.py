import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, BatchNormalization, Flatten, Dense, Lambda, tf

def dl_model():     

        input_layer = Input(shape=(32, 32, 3))
        
        # Branch Path
        branch_path = AveragePooling2D(pool_size=(8, 8))(input_layer)  

        # Main Path
        x = input_layer
        
        # Block 1
        x = Lambda(lambda x: tf.split(x, 3, axis=3))(x)  # Split channels
        x = [Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(xi) for xi in x]
        x = Concatenate()(x)  # Concatenate outputs

        # Block 2
        x = Lambda(lambda x: tf.keras.backend.shape(x)[1:3])(x) # Get height and width
        x = tf.reshape(x, (-1, x, x, 3, 64))
        x = tf.transpose(x, perm=[0, 1, 2, 4, 3])
        x = tf.reshape(x, (-1, x, x, 64, 3))

        # Block 3
        x = Conv2D(filters=128, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu', depth_wise=True)(x)

        # Concatenate Main and Branch Paths
        x = Concatenate()([x, branch_path]) 

        # Fully Connected Layers
        x = Flatten()(x)
        x = Dense(units=128, activation='relu')(x)
        output_layer = Dense(units=10, activation='softmax')(x)  

        model = keras.Model(inputs=input_layer, outputs=output_layer)

        return model