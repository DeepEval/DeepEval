from keras.models import Model
from keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, Dense, GlobalAveragePooling2D, Reshape, Multiply, Conv2D, Flatten, Dense

def dl_model():
    # Input layer
    inputs = Input(shape=(32, 32, 3))

    # Increase the dimensionality of the input channels
    x = Conv2D(32, (1, 1), activation='relu')(inputs)

    # Extract initial features using a 3x3 depthwise separable convolution
    x = Conv2D(32, (3, 3), activation='relu', depthwise=True)(x)
    x = Conv2D(32, (1, 1), activation='relu')(x)

    # Compute channel attention weights
    x = GlobalAveragePooling2D()(x)
    x = Dense(512, activation='relu')(x)
    attention_weights = Dense(32, activation='softmax')(x)

    # Reshape attention weights to match the initial features
    attention_weights = Reshape((32, 2, 2))(attention_weights)

    # Multiply initial features with attention weights
    x = multiply([x, attention_weights])

    # Reduce dimensionality with a 1x1 convolution
    x = Conv2D(64, (1, 1), activation='relu')(x)

    # Flatten and pass through a fully connected layer for classification
    x = Flatten()(x)
    outputs = Dense(10, activation='softmax')(x)

    # Construct the model
    model = Model(inputs=inputs, outputs=outputs)
    
    return model

# Get the constructed model
model = dl_model()
model.summary()