import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, Lambda, GlobalAveragePooling2D, Dense

def dl_model():
    
    input_layer = Input(shape=(32, 32, 3))
    
    # First Block
    x = Lambda(lambda x: tf.split(x, 3, axis=2))(input_layer)
    
    # Branch 1
    branch1 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(x[0])
    # Branch 2
    branch2 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(x[1])
    # Branch 3
    branch3 = Conv2D(filters=32, kernel_size=(5, 5), strides=(1, 1), padding='same', activation='relu')(x[2])

    # Concatenate branches
    x = Concatenate()([branch1, branch2, branch3])

    # Second Block
    
    # Branch 1
    branch4 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(x)
    
    # Branch 2
    branch5 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(x)
    branch5 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(branch5)
    branch5 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(branch5)
    
    # Branch 3
    branch6 = MaxPooling2D(pool_size=(2, 2), strides=2, padding='same')(x)
    
    # Concatenate branches
    x = Concatenate()([branch4, branch5, branch6])

    # Global Average Pooling
    x = GlobalAveragePooling2D()(x)
    
    # Output Layer
    output_layer = Dense(units=10, activation='softmax')(x)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model