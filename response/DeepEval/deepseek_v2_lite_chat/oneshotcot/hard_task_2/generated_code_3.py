import tensorflow as tf
from tensorflow.keras.layers import Input, Lambda, Conv2D, MaxPooling2D, Add, Flatten, Dense
from tensorflow.keras.models import Model

def dl_model():
    # Input layer
    input_layer = Input(shape=(32, 32, 3))
    
    # Split input into three groups along channel dimension
    channel_splits = Lambda(lambda x: tf.split(x, 3, axis=-1))(input_layer)
    
    # Convolutional layers for each group
    conv1 = Conv2D(filters=32, kernel_size=(1, 1), padding='same')(channel_splits[0])
    conv2 = Conv2D(filters=32, kernel_size=(3, 3), padding='same')(channel_splits[1])
    conv3 = Conv2D(filters=32, kernel_size=(1, 1), padding='same')(channel_splits[2])
    
    # Add the convolutional layers
    main_path = Add()([conv1, conv2, conv3])
    
    # Concatenate the main path with the original input
    fused_layer = Add()([main_path, channel_splits[0]])
    
    # Flatten and fully connected layers
    flatten = Flatten()(fused_layer)
    dense1 = Dense(units=128, activation='relu')(flatten)
    dense2 = Dense(units=64, activation='relu')(dense1)
    output_layer = Dense(units=10, activation='softmax')(dense2)
    
    # Build the model
    model = Model(inputs=input_layer, outputs=output_layer)
    
    return model

# Create the model
model = dl_model()
model.summary()