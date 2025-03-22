from keras.layers import Input, Conv2D, GlobalAveragePooling2D, Dense, Multiply, Add, Flatten
from keras.models import Model
from keras.datasets import cifar10
from keras.utils import to_categorical

def dl_model():
    # Input layer
    input_layer = Input(shape=(32, 32, 3))

    # Main Path
    # Global average pooling
    x = GlobalAveragePooling2D()(input_layer)
    
    # Fully connected layers to generate weights
    x = Dense(64, activation='relu')(x)
    x = Dense(3, activation='sigmoid')(x)  # Assuming 3 channels for the final adjustment (CIFAR-10)

    # Reshape weights to match the input layer's shape (3 channels)
    x = Dense(32 * 32 * 3, activation='relu')(x)
    weights = Reshape((32, 32, 3))(x)
    
    # Element-wise multiplication with original feature map
    weighted_feature_map = Multiply()([input_layer, weights])

    # Branch Path
    # Convolution to match the channels of the input layer
    branch_output = Conv2D(3, (3, 3), padding='same', activation='relu')(input_layer)

    # Adding outputs from both paths
    combined_output = Add()([weighted_feature_map, branch_output])

    # Fully connected layers for classification
    x = Flatten()(combined_output)
    x = Dense(512, activation='relu')(x)
    x = Dense(256, activation='relu')(x)
    output_layer = Dense(10, activation='softmax')(x)

    # Construct the model
    model = Model(inputs=input_layer, outputs=output_layer)

    return model

# Example usage:
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)
model = dl_model()
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(x_train, y_train, epochs=10, batch_size=64, validation_data=(x_test, y_test))