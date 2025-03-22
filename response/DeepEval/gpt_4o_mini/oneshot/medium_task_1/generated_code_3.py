import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Add, Flatten, Dense
from keras.models import Model

def dl_model():
    input_layer = Input(shape=(32, 32, 3))  # CIFAR-10 images are 32x32 RGB images

    # First convolutional layer
    conv1 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_layer)
    
    # Second convolutional layer
    conv2 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(conv1)
    
    # Max pooling layer
    max_pooling = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid')(conv2)

    # Adding the output features directly with the input layer
    added_features = Add()([max_pooling, input_layer])

    # Flatten the features
    flatten_layer = Flatten()(added_features)

    # Fully connected layers
    dense1 = Dense(units=128, activation='relu')(flatten_layer)
    dense2 = Dense(units=64, activation='relu')(dense1)
    
    # Output layer for 10 classes
    output_layer = Dense(units=10, activation='softmax')(dense2)

    # Construct the model
    model = Model(inputs=input_layer, outputs=output_layer)

    return model

# You can create the model by calling the function
model = dl_model()
model.summary()