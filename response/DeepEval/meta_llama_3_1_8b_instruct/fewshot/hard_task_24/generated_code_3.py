import keras
from keras.layers import Input, Conv2D, Concatenate, UpSampling2D, MaxPooling2D, Dense
from keras.layers import Lambda, Reshape
from tensorflow import keras as tf

def dl_model():
    input_layer = Input(shape=(32, 32, 3))
    
    # Branch 1: extract local features through a 3x3 convolutional layer
    conv1 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_layer)
    
    # Branch 2: max pooling and upsampling
    maxpool = MaxPooling2D(pool_size=(2, 2), strides=2, padding='same')(input_layer)
    conv2 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(maxpool)
    upsample = UpSampling2D(size=(2, 2))(conv2)
    
    # Branch 3: max pooling and upsampling
    maxpool = MaxPooling2D(pool_size=(2, 2), strides=2, padding='same')(input_layer)
    conv3 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(maxpool)
    upsample = UpSampling2D(size=(2, 2))(conv3)
    
    # Fuse the outputs of all branches
    fusion = Concatenate()([conv1, upsample, upsample])
    
    # Apply a 1x1 convolutional layer
    conv_fusion = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(fusion)
    
    # Flatten and pass through three fully connected layers
    flatten = tf.keras.layers.Reshape(target_shape=(32*32*64))(conv_fusion)
    dense1 = Dense(units=128, activation='relu')(flatten)
    dense2 = Dense(units=64, activation='relu')(dense1)
    output_layer = Dense(units=10, activation='softmax')(dense2)
    
    model = keras.Model(inputs=input_layer, outputs=output_layer)
    
    return model

def main():
    model = dl_model()
    model.summary()
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

if __name__ == "__main__":
    main()