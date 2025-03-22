import keras
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Input, concatenate
from keras.models import Model
from keras.layers import LeakyReLU, BatchNormalization, Add

def dl_model():
    # Define the input shape
    input_shape = (32, 32, 3)  # CIFAR-10 images are 32x32

    # Input layer
    inputs = Input(shape=input_shape)

    # Convolution layer to increase channels
    conv1 = Conv2D(32, (1, 1), activation='relu')(inputs)

    # Depthwise separable convolution
    depthwise = Conv2D(32, (3, 3), activation='relu', depth_kwouts=1)(conv1)
    pointwise = Conv2D(64, (1, 1), activation='relu')(depthwise)

    # Channel attention module
    ap = Add()([pointwise, depthwise])
    ap = LeakyReLU()(ap)
    gp = GlobalAveragePooling2D()(ap)

    # Fully connected layers for channel attention
    fc1 = Dense(256, activation='relu')(gp)
    fc2 = Dense(64, activation='relu')(fc1)

    # Reshape to match the initial features and multiply
    weights = Dense(64)(fc2)
    weights = Dense(64)(weights)
    attention_weights = Lambda(lambda x: tf.reshape(x, (-1, 64)))(weights)
    attention_output = tf.matmul(pointwise, attention_weights)

    # Add attention weights to the initial features
    outputs = Add()([ap, attention_output])

    # Reduce dimensionality with another convolution
    output = Conv2D(1, (1, 1), activation='relu')(outputs)

    # Flatten and fully connected layers for classification
    flat = Flatten()(output)
    fc = Dense(10, activation='softmax')(flat)

    # Create the model
    model = Model(inputs=inputs, outputs=fc)

    return model

# Create the model
model = dl_model()
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])