import keras
from keras.layers import Input, Lambda, Conv2D, MaxPooling2D, Concatenate, BatchNormalization, Flatten, Dense
from keras.models import Model

def dl_model():
    # Input layer
    input_layer = Input(shape=(32, 32, 3))
    
    # Split the input into three groups along the channel dimension
    channel_splits = Lambda(lambda x: tf.split(x, 3, axis=-1))(input_layer)
    
    # Convolutional layers
    conv1 = Conv2D(filters=64, kernel_size=(1, 1), padding='same')(channel_splits[0])
    conv2 = Conv2D(filters=64, kernel_size=(3, 3), padding='same')(channel_splits[1])
    conv3 = Conv2D(filters=64, kernel_size=(5, 5), padding='same')(channel_splits[2])
    
    # MaxPooling layers
    pool1 = MaxPooling2D(pool_size=(1, 1), padding='valid')(conv1)
    pool2 = MaxPooling2D(pool_size=(1, 1), padding='valid')(conv2)
    pool3 = MaxPooling2D(pool_size=(1, 1), padding='valid')(conv3)
    
    # Concatenate the outputs from the three groups
    concat = Concatenate(axis=-1)([pool1, pool2, pool3])
    
    # Batch normalization and Flatten layers
    bn = BatchNormalization()(concat)
    flat = Flatten()(bn)
    
    # Fully connected layers
    dense1 = Dense(units=128, activation='relu')(flat)
    dense2 = Dense(units=64, activation='relu')(dense1)
    
    # Output layer
    output_layer = Dense(units=10, activation='softmax')(dense2)
    
    # Construct the model
    model = Model(inputs=input_layer, outputs=output_layer)
    
    return model

# Instantiate and compile the model
model = dl_model()
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])