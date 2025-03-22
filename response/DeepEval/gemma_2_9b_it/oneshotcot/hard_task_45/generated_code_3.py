import keras
from keras.layers import Input, Conv2D, Lambda, Concatenate, MaxPooling2D, BatchNormalization, Flatten, Dense

def dl_model():     

    input_layer = Input(shape=(32, 32, 3))

    # First Block: Depthwise Separable Convolutions
    x = Lambda(lambda x: tf.split(x, num_or_size_splits=3, axis=-1))(input_layer)
    
    # Group 1
    x1 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(x[0])
    
    # Group 2
    x2 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(x[1])
    
    # Group 3
    x3 = Conv2D(filters=32, kernel_size=(5, 5), strides=(1, 1), padding='same', activation='relu')(x[2])

    x = Concatenate()([x1, x2, x3])  

    # Second Block: Feature Extraction Branches
    branch1 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(x)
    branch1 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(branch1)
    branch1 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(branch1)
    branch1 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(branch1)
    
    branch2 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(x)
    branch2 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(branch2)
    
    branch3 = MaxPooling2D(pool_size=(2, 2), strides=2, padding='same')(x)
    branch3 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(branch3) 
    
    x = Concatenate()([branch1, branch2, branch3])

    # Flatten and Fully Connected
    x = Flatten()(x)
    output_layer = Dense(units=10, activation='softmax')(x)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model