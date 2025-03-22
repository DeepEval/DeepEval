import keras
from keras.layers import Input, Lambda, Conv2D, MaxPool2D, Flatten, Dense, Add, Concatenate
from keras.models import Model

def dl_model():
    # Input layer
    input_layer = Input(shape=(32, 32, 3))
    
    # Split into three channel groups
    split1 = Lambda(lambda x: tf.split(x, num_or_size_splits=3, axis=1))(input_layer)
    split2 = Lambda(lambda x: tf.split(x, num_or_size_splits=3, axis=1))(split1[0])
    split3 = Lambda(lambda x: tf.split(x, num_or_size_splits=3, axis=1))(split1[1])
    
    # Process each channel group with separable convolutions
    conv1_1 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same')(split1[0])
    conv1_2 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same')(split1[1])
    conv1_3 = Conv2D(filters=32, kernel_size=(5, 5), strides=(1, 1), padding='same')(split1[2])
    
    conv2_1 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same')(split2[0])
    conv2_2 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same')(split2[1])
    conv2_3 = Conv2D(filters=64, kernel_size=(5, 5), strides=(1, 1), padding='same')(split2[2])
    
    conv3_1 = Conv2D(filters=128, kernel_size=(1, 1), strides=(1, 1), padding='same')(split3[0])
    conv3_2 = Conv2D(filters=128, kernel_size=(3, 3), strides=(1, 1), padding='same')(split3[1])
    conv3_3 = Conv2D(filters=128, kernel_size=(5, 5), strides=(1, 1), padding='same')(split3[2])
    
    # Pooling
    pool1 = MaxPool2D(pool_size=(2, 2), strides=(2, 2))(conv1_1)
    pool2 = MaxPool2D(pool_size=(2, 2), strides=(2, 2))(conv2_1)
    pool3 = MaxPool2D(pool_size=(2, 2), strides=(2, 2))(conv3_1)
    
    # Concatenate the outputs
    concat = Concatenate(axis=-1)([pool1, pool2, pool3])
    
    # Flatten and pass through fully connected layers
    flatten = Flatten()(concat)
    dense1 = Dense(units=128, activation='relu')(flatten)
    dense2 = Dense(units=64, activation='relu')(dense1)
    output_layer = Dense(units=10, activation='softmax')(dense2)
    
    # Model
    model = Model(inputs=input_layer, outputs=output_layer)
    
    return model

# Build the model
model = dl_model()
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])