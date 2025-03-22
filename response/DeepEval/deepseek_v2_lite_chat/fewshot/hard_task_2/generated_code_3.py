import keras
from keras.layers import Input, Lambda, Conv2D, Add, Flatten, Dense

def dl_model():
    # Input layer
    input_layer = Input(shape=(32, 32, 3))
    
    # Split the input into three groups
    split1 = Lambda(lambda x: tf.split(x, num_or_size_splits=3, axis=1))(input_layer)
    
    # Convolution layers for each group
    conv1 = Conv2D(filters=16, kernel_size=(1, 1), padding='same')(split1[0])  # 1x1 conv
    conv2 = Conv2D(filters=16, kernel_size=(3, 3), padding='same')(split1[1])  # 3x3 conv
    conv3 = Conv2D(filters=16, kernel_size=(1, 1), padding='same')(split1[2])  # 1x1 conv
    
    # Add all three groups and pass through a fully connected layer
    add_layer = Add()([conv1, conv2, conv3])
    flatten_layer = Flatten()(add_layer)
    
    # Fully connected layer for classification
    output_layer = Dense(units=10, activation='softmax')(flatten_layer)
    
    # Model construction
    model = keras.Model(inputs=input_layer, outputs=output_layer)
    
    return model

# Create the model
model = dl_model()
model.summary()