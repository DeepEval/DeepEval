from keras.models import Model
from keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from keras.layers import LeakyReLU
from keras.models import model_from_json

def dl_model():
    # Input layer
    input_layer = Input(shape=(224, 224, 3))
    
    # Feature extraction layers
    x = Conv2D(32, (3, 3), activation='relu')(input_layer)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    
    x = Conv2D(64, (3, 3), activation='relu')(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    
    x = Conv2D(128, (3, 3), activation='relu')(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    
    # Flattening
    x = Flatten()(x)
    
    # Fully connected layers
    x = Dense(512, activation='relu')(x)
    x = Dropout(0.5)(x)  # Dropout to prevent overfitting
    
    output = Dense(1000, activation='softmax')(x)  # 1000 classes
    
    # Model construction
    model = Model(inputs=input_layer, outputs=output)
    
    return model

# Example usage:
if __name__ == "__main__":
    model = dl_model()
    model.summary()  # Print a summary of the model
    # Save the model
    model_json = model.to_json()
    with open("model.json", "w") as json_file:
        json_file.write(model_json)
    # Compile the model
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])