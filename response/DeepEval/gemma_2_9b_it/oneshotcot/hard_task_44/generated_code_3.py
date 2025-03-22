import keras
from keras.layers import Input, Conv2D, Lambda, MaxPooling2D, Concatenate, Dropout, Flatten, Dense

def dl_model():
    
    input_layer = Input(shape=(32, 32, 3))
    
    # Block 1
    x = Lambda(lambda x: tf.split(x, 3, axis=2))(input_layer)
    
    # Feature extraction with varying kernel sizes
    path1 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(x[0])
    path2 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(x[1])
    path3 = Conv2D(filters=32, kernel_size=(5, 5), strides=(1, 1), padding='same', activation='relu')(x[2])
    
    # Concatenate outputs from three paths
    x = Concatenate()([path1, path2, path3])
    
    # Apply dropout
    x = Dropout(0.25)(x)
    
    # Block 2
    x = Lambda(lambda x: tf.split(x, 4, axis=2))(x)
    
    # Branch 1
    branch1 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(x[0])
    branch1 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(branch1)
    
    # Branch 2
    branch2 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(x[1])
    branch2 = Conv2D(filters=32, kernel_size=(5, 5), strides=(1, 1), padding='same', activation='relu')(branch2)
    
    # Branch 3
    branch3 = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='same')(x[2])
    branch3 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(branch3)
    
    # Concatenate outputs from all branches
    x = Concatenate()([branch1, branch2, branch3])
    
    # Flatten and output layer
    x = Flatten()(x)
    output_layer = Dense(units=10, activation='softmax')(x)
    
    model = keras.Model(inputs=input_layer, outputs=output_layer)
    return model