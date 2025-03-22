import keras
from keras.datasets import mnist
from keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, Concatenate, Dense, Flatten
from keras.models import Model
from keras.layers import LayerNormalization

def dl_model():
    # Input layers for the two pathways
    input1 = Input(shape=(28, 28, 1))
    input2 = Input(shape=(28, 28, 1))

    # Encoder
    def encoder(input_layer):
        x = Conv2D(64, (3, 3), activation='relu', padding='same')(input_layer)
        x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
        x = MaxPooling2D((2, 2), padding='same')(x)
        x = Conv2D(128, (3, 3), activation='relu', padding='same')(x)
        x = Conv2D(128, (3, 3), activation='relu', padding='same')(x)
        x = MaxPooling2D((2, 2), padding='same')(x)
        return x

    encoded1 = encoder(input1)
    encoded2 = encoder(input2)

    # Decoder
    def decoder(input_layer):
        x = Conv2D(128, (3, 3), activation='relu', padding='same')(input_layer)
        x = Conv2D(128, (3, 3), activation='relu', padding='same')(x)
        x = UpSampling2D((2, 2))(x)
        x = Conv2D(128, (3, 3), activation='relu')(x)
        x = Conv2D(128, (3, 3), activation='relu')(x)
        x = UpSampling2D((2, 2))(x)
        return x

    decoded1 = decoder(encoded1)
    decoded2 = decoder(encoded2)

    # Batch normalization and ReLU activation
    x1 = LayerNormalization()(decoded1)
    x2 = LayerNormalization()(decoded2)
    x1 = keras.activations.relu(x1)
    x2 = keras.activations.relu(x2)

    # Concatenate features from both pathways
    x = Concatenate()([x1, x2])

    # Fully connected layers
    x = Flatten()(x)
    output = Dense(10, activation='softmax')(x)

    # Model
    model = Model(inputs=[input1, input2], outputs=output)

    return model

# Create the model
model = dl_model()

# Print the model summary
model.summary()