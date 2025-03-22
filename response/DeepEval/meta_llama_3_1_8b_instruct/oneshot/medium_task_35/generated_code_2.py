import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, BatchNormalization, Flatten, Dense, Dropout, UpSampling2D, Conv2DTranspose
from keras.optimizers import Adam
from keras.losses import CategoricalCrossentropy
from keras.metrics import CategoricalAccuracy
import tensorflow as tf

def dl_model():     

    input_layer = Input(shape=(32, 32, 3))
    
    # Stage 1
    conv1 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_layer)
    max_pooling1 = MaxPooling2D(pool_size=(2, 2), strides=2, padding='same')(conv1)
    
    # Stage 2
    conv2 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(max_pooling1)
    max_pooling2 = MaxPooling2D(pool_size=(2, 2), strides=2, padding='same')(conv2)
    
    # Additional convolutional layers for feature extraction
    conv3 = Conv2D(filters=128, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(max_pooling2)
    drop = Dropout(0.2)(conv3)
    conv4 = Conv2D(filters=128, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(drop)
    
    # Decoder
    up1 = Conv2DTranspose(filters=128, kernel_size=(2, 2), strides=(2, 2), padding='same', activation='relu')(conv4)
    skip_conn1 = Concatenate()([up1, conv3])
    drop_conn1 = Dropout(0.2)(skip_conn1)
    conv5 = Conv2D(filters=128, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(drop_conn1)
    
    up2 = Conv2DTranspose(filters=64, kernel_size=(2, 2), strides=(2, 2), padding='same', activation='relu')(conv5)
    skip_conn2 = Concatenate()([up2, conv2])
    drop_conn2 = Dropout(0.2)(skip_conn2)
    conv6 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(drop_conn2)
    
    up3 = Conv2DTranspose(filters=32, kernel_size=(2, 2), strides=(2, 2), padding='same', activation='relu')(conv6)
    skip_conn3 = Concatenate()([up3, conv1])
    drop_conn3 = Dropout(0.2)(skip_conn3)
    conv7 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(drop_conn3)
    
    # Output layer
    output_layer = Conv2D(filters=10, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='softmax')(conv7)
    
    # Reshape the output for the categorical cross-entropy loss
    output_layer = tf.reshape(output_layer, (-1, 10))
    
    model = keras.Model(inputs=input_layer, outputs=output_layer)
    model.compile(optimizer=Adam(lr=0.001), 
                  loss=CategoricalCrossentropy(from_logits=False), 
                  metrics=[CategoricalAccuracy()])
    
    return model