import keras
from keras.models import Model
from keras.layers import Input, Lambda, Conv2D, AveragePooling2D, Concatenate, Flatten, Dense

def dl_model():
    # Input layer
    input_layer = Input(shape=(32, 32, 3))
    
    # Split the input into three groups along the channel dimension
    group1 = Lambda(lambda x: x[:, :, :, 0])(input_layer)
    group2 = Lambda(lambda x: x[:, :, :, 1:3])(input_layer)
    group3 = Lambda(lambda x: x[:, :, :, 2:])(input_layer)
    
    # Convolutional layers for each group
    conv1 = Conv2D(filters=32, kernel_size=(1, 1), padding='same')(group1)
    conv2 = Conv2D(filters=32, kernel_size=(3, 3), padding='same')(group2)
    conv3 = Conv2D(filters=32, kernel_size=(5, 5), padding='same')(group3)
    
    # Pooling layers for each group
    pool1 = AveragePooling2D(pool_size=(2, 2))(conv1)
    pool2 = AveragePooling2D(pool_size=(2, 2))(conv2)
    pool3 = AveragePooling2D(pool_size=(2, 2))(conv3)
    
    # Concatenate the outputs from the three groups
    concatenated = Concatenate(axis=-1)([pool1, pool2, pool3])
    
    # Flatten the concatenated features and pass through fully connected layers
    flattened = Flatten()(concatenated)
    dense1 = Dense(units=128, activation='relu')(flattened)
    dense2 = Dense(units=64, activation='relu')(dense1)
    output_layer = Dense(units=10, activation='softmax')(dense2)
    
    # Return the constructed model
    model = Model(inputs=input_layer, outputs=output_layer)
    return model

# Create the model
model = dl_model()

# Display the model summary
model.summary()