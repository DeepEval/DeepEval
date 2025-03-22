import keras
from keras.layers import Input, Lambda, Conv2D, MaxPool2D, Add, Flatten, Dense

def dl_model():
    # Input layer
    input_layer = Input(shape=(32, 32, 3))
    
    # Split input into three groups
    split1 = Lambda(lambda x: tf.split(x, num_or_size_splits=3, axis=-1))(input_layer)
    
    # Process each group through convolutions
    def conv_block(group):
        conv1 = Conv2D(filters=64, kernel_size=(1, 1), padding='same', activation='relu')(group[0])
        conv2 = Conv2D(filters=64, kernel_size=(3, 3), padding='same', activation='relu')(group[1])
        conv3 = Conv2D(filters=64, kernel_size=(1, 1), padding='same', activation='relu')(group[2])
        return conv3  # Convolutional output from the third group is used
    
    # Extract features from each group
    features1 = conv_block(split1)
    features2 = conv_block(split1)
    features3 = conv_block(split1)
    
    # Merge features with original input and add back the original input
    merge1 = Add()([input_layer, features1])
    merge2 = Add()([merge1, features2])
    merge3 = Add()([merge2, features3])
    
    # Flatten and classify
    flatten = Flatten()(merge3)
    output_layer = Dense(units=10, activation='softmax')(flatten)
    
    # Model architecture
    model = keras.Model(inputs=input_layer, outputs=output_layer)
    
    return model

# Build and compile the model
model = dl_model()
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])