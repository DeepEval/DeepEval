import keras
from keras.layers import Input, Lambda, Conv2D, MaxPooling2D, Concatenate, BatchNormalization, Flatten, Dense
from keras.models import Model

def dl_model():
    # Define the input shape (32x32x3) for CIFAR-10 images
    input_layer = Input(shape=(32, 32, 3))
    
    # Split the input into three groups along the last dimension
    group1, group2, group3 = Lambda(lambda x: tf.split(x, 3, axis=-1))(input_layer)
    
    # Define depthwise separable convolutional layers for each group
    # Kernel sizes: 1x1, 3x3, 5x5
    conv1 = Conv2D(filters=32, kernel_size=1, padding='same')(group1)
    conv2 = Conv2D(filters=32, kernel_size=3, padding='same')(group2)
    conv3 = Conv2D(filters=32, kernel_size=5, padding='same')(group3)
    
    # MaxPooling layers
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
    
    # Concatenate the outputs from the three groups
    fused_features = Concatenate()(pool1, pool2, pool3)
    
    # Flatten and pass through fully connected layers
    flatten_layer = Flatten()(fused_features)
    dense1 = Dense(units=128, activation='relu')(flatten_layer)
    dense2 = Dense(units=64, activation='relu')(dense1)
    output_layer = Dense(units=10, activation='softmax')(dense2)
    
    # Construct the model
    model = Model(inputs=input_layer, outputs=output_layer)
    
    return model

# Instantiate the model
model = dl_model()
model.summary()