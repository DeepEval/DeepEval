import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Add, Flatten, Concatenate, Lambda, Dense

def dl_model():
    # Input layer
    input_layer = Input(shape=(32, 32, 3))
    
    # First block: three groups with depthwise separable convolutional layers
    def block1(input_tensor):
        split1 = Lambda(lambda x: tf.split(x, num_or_size_splits=3, axis=-1))(input_tensor)
        conv1_1 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(split1[0])
        conv1_2 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(split1[1])
        conv1_3 = Conv2D(filters=64, kernel_size=(5, 5), strides=(1, 1), padding='same', activation='relu')(split1[2])
        pool1 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid')(conv1_3)
        pooled1 = Flatten()(pool1)
        
        return Concatenate()([pooled1, conv1_1, conv1_2, conv1_3])
    
    # Second block: multiple branches for feature extraction
    def block2(input_tensor):
        split2 = Lambda(lambda x: tf.split(x, num_or_size_splits=2, axis=-1))(input_tensor)
        conv2_1 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(split2[0])
        conv2_2 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(split2[1])
        conv2_3 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(split2[1])
        pool2 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid')(conv2_3)
        pooled2 = Flatten()(pool2)
        maxpool = MaxPooling2D(pool_size=(3, 3), strides=(3, 3), padding='valid')(conv2_1)
        pooledmax = Flatten()(maxpool)
        
        return Concatenate()([pooled2, conv2_1, conv2_2, conv2_3, pooledmax, pooled2])
    
    # Construct the model
    model = keras.Model(inputs=input_layer, outputs=block2(block1(input_tensor=input_layer)))
    
    return model

# Build the model
model = dl_model()
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])