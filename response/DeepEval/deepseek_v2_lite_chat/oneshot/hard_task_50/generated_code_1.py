import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, Lambda, Concatenate, Add
from keras.models import Model

def dl_model():
    # First block: three max pooling layers with different scales
    input_layer = Input(shape=(32, 32, 3))  # CIFAR-10 images are 32x32
    x = MaxPooling2D(pool_size=(1, 1), strides=(1, 1), padding='valid')(input_layer)
    x = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid')(x)
    x = MaxPooling2D(pool_size=(4, 4), strides=(4, 4), padding='valid')(x)
    x = Flatten()(x)  # Flatten the output from max pooling
    dropout = keras.layers.Dropout(0.5)(x)  # Dropout to mitigate overfitting
    
    # Second block: processing the output from the first block
    split1 = Lambda(lambda x: keras.layers.split(x, 2))(dropout)
    split2 = Lambda(lambda x: keras.layers.split(x, 2))(dropout)
    split3 = Lambda(lambda x: keras.layers.split(x, 2))(dropout)
    split4 = Lambda(lambda x: keras.layers.split(x, 2))(dropout)
    
    # Separate convolutional layers for each group
    conv1 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(split1[0])
    conv2 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(split2[0])
    conv3 = Conv2D(filters=64, kernel_size=(5, 5), strides=(1, 1), padding='same', activation='relu')(split3[0])
    conv4 = Conv2D(filters=64, kernel_size=(7, 7), strides=(1, 1), padding='same', activation='relu')(split4[0])
    
    # Concatenate the outputs from the separable convolutions
    concat = Concatenate()( [conv1, conv2, conv3, conv4] )
    
    # Fully connected layer for final classification
    dense1 = Dense(units=256, activation='relu')(concat)
    dense2 = Dense(units=128, activation='relu')(dense1)
    output_layer = Dense(units=10, activation='softmax')(dense2)

    model = Model(inputs=input_layer, outputs=output_layer)

    return model