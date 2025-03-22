import keras
from keras.layers import Input, Conv2D, BatchNormalization, ReLU, Add, AveragePooling2D, Flatten, Dense
from keras.models import Model

def dl_model():
    # Input layer
    input_layer = Input(shape=(32, 32, 3))

    # Initial convolutional layer
    conv1 = Conv2D(filters=16, kernel_size=(3, 3), strides=(1, 1), padding='same')(input_layer)
    bn1 = BatchNormalization()(conv1)
    relu1 = ReLU()(bn1)

    # Basic block 1
    conv2 = Conv2D(filters=16, kernel_size=(3, 3), strides=(1, 1), padding='same')(relu1)
    bn2 = BatchNormalization()(conv2)
    relu2 = ReLU()(bn2)
    add1 = Add()([relu1, relu2])

    # Branch block 1
    branch1 = Conv2D(filters=16, kernel_size=(3, 3), strides=(1, 1), padding='same')(add1)

    # Basic block 2
    conv3 = Conv2D(filters=16, kernel_size=(3, 3), strides=(1, 1), padding='same')(add1)
    bn3 = BatchNormalization()(conv3)
    relu3 = ReLU()(bn3)
    add2 = Add()([relu2, relu3])

    # Branch block 2
    branch2 = Conv2D(filters=16, kernel_size=(3, 3), strides=(1, 1), padding='same')(add2)

    # Feature fusion
    add3 = Add()([branch1, branch2])

    # Average pooling
    avg_pool = AveragePooling2D()(add3)

    # Flattening
    flatten = Flatten()(avg_pool)

    # Fully connected layers
    dense1 = Dense(units=128, activation='relu')(flatten)
    dense2 = Dense(units=64, activation='relu')(dense1)

    # Output layer
    output_layer = Dense(units=10, activation='softmax')(dense2)

    # Model construction
    model = Model(inputs=input_layer, outputs=output_layer)

    return model

# Print the model summary
model = dl_model()
model.summary()