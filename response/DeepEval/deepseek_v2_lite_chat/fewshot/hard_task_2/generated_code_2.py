import keras
from keras.layers import Input, Lambda, Conv2D, MaxPooling2D, Add, Flatten, Dense
from keras.models import Model

def dl_model():
    # Input layer
    input_layer = Input(shape=(32, 32, 3))
    
    # Split input into three groups
    split1 = Lambda(lambda x: tf.split(x, num_or_size_splits=3, axis=1))(input_layer)
    
    # Process each group
    conv1_1 = Conv2D(filters=32, kernel_size=(1, 1), activation='relu')(split1[0])
    conv1_2 = Conv2D(filters=32, kernel_size=(3, 3), activation='relu')(split1[1])
    conv1_3 = Conv2D(filters=32, kernel_size=(1, 1), activation='relu')(split1[2])
    
    # Combine results
    add1 = Add()([conv1_1, conv1_2, conv1_3])
    pool1 = MaxPooling2D(pool_size=(2, 2))(add1)
    
    # Flatten and fully connected layers
    flatten = Flatten()(pool1)
    dense1 = Dense(units=64, activation='relu')(flatten)
    dense2 = Dense(units=10, activation='softmax')(dense1)
    
    # Return the constructed model
    model = Model(inputs=input_layer, outputs=dense2)
    return model

# Build and compile the model
model = dl_model()
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])