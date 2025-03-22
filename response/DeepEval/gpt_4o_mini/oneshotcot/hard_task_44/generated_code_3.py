import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, Dropout, Flatten, Dense, Lambda
import tensorflow as tf

def dl_model():
    input_layer = Input(shape=(32, 32, 3))  # CIFAR-10 images are 32x32 with 3 color channels
    
    # Block 1
    split_outputs = Lambda(lambda x: tf.split(x, num_or_size_splits=3, axis=-1))(input_layer)
    
    conv1 = Conv2D(filters=32, kernel_size=(1, 1), padding='same', activation='relu')(split_outputs[0])
    conv2 = Conv2D(filters=32, kernel_size=(3, 3), padding='same', activation='relu')(split_outputs[1])
    conv3 = Conv2D(filters=32, kernel_size=(5, 5), padding='same', activation='relu')(split_outputs[2])
    
    # Applying dropout to reduce overfitting
    dropout1 = Dropout(rate=0.5)(Concatenate()([conv1, conv2, conv3]))
    
    # Block 2
    branch1 = Conv2D(filters=64, kernel_size=(1, 1), padding='same', activation='relu')(dropout1)
    branch2 = Conv2D(filters=64, kernel_size=(1, 1), padding='same', activation='relu')(Conv2D(filters=64, kernel_size=(3, 3), padding='same', activation='relu')(dropout1))
    branch3 = Conv2D(filters=64, kernel_size=(1, 1), padding='same', activation='relu')(Conv2D(filters=64, kernel_size=(5, 5), padding='same', activation='relu')(dropout1))
    branch4 = Concatenate()([
        MaxPooling2D(pool_size=(3, 3), strides=(1, 1), padding='same')(dropout1), 
        Conv2D(filters=64, kernel_size=(1, 1), padding='same', activation='relu')(dropout1)
    ])
    
    # Concatenate outputs of all branches in Block 2
    block2_output = Concatenate()([branch1, branch2, branch3, branch4])
    
    # Final layers
    flatten_layer = Flatten()(block2_output)
    dense_layer = Dense(units=128, activation='relu')(flatten_layer)
    output_layer = Dense(units=10, activation='softmax')(dense_layer)

    # Constructing the model
    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model