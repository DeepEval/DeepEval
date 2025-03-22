import keras
from keras.layers import Input, Conv2D, Add, concatenate, Flatten, Reshape, Permute, Lambda, Dense
from keras.models import Model

def dl_model():
    # Input layer
    input_A = Input(shape=(28, 28, 1))  # Assuming the images are 28x28 grayscale
    input_B = Input(shape=(28, 28, 1))  # Assuming the images are 28x28 grayscale

    # Block 1
    # 1x1 convolution
    x1 = Conv2D(32, (1, 1), padding='same')(input_A)
    # 3x3 depthwise separable convolution
    x1 = Conv2D(64, (3, 3), padding='same', activation='relu')(x1)
    # 1x1 convolution
    x1 = Conv2D(128, (1, 1), padding='same')(x1)
    
    # Block 2
    # 3x3 depthwise separable convolution
    x2 = Conv2D(32, (3, 3), padding='same', activation='relu')(input_B)
    # 1x1 convolution
    x2 = Conv2D(64, (1, 1), padding='same')(x2)

    # Concatenate features
    z = Add()([x1, x2])

    # Block 3
    # Reshape to four groups
    z = Reshape((-1, z.shape[1]*z.shape[2]))(z)
    z = Lambda(lambda x: tf.split(x, num_or_size_splits=4, axis=1))(z)
    
    # Swap dimensions
    z = Permute((2, 3, 1))(z)
    # Flatten and feed into dense layers
    z = Flatten()(z)
    z = Dense(256, activation='relu')(z)

    # Output layer
    output = Dense(10, activation='softmax')(z)  # Assuming 10 classes for MNIST

    # Define the model
    model = Model(inputs=[input_A, input_B], outputs=output)

    return model

# Create the model
model = dl_model()
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])