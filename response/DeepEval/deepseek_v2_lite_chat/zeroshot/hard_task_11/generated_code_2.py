from keras.layers import Input, Conv2D, Add, concatenate, Dense, GlobalAveragePooling2D
from keras.models import Model

def dl_model():
    # Input layer
    inputs = Input(shape=(32, 32, 3))
    
    # Main pathway
    x = inputs
    for _ in range(4):  # four branches: 1x1, 1x3, 3x1, input to branch
        x = Conv2D(filters=16, kernel_size=(1, 1), activation='relu')(x)
        x = Conv2D(filters=16, kernel_size=(1, 3), activation='relu')(x)
        x = Conv2D(filters=16, kernel_size=(3, 1), activation='relu')(x)
        x = Conv2D(filters=16, kernel_size=(1, 1), activation='relu')(x)
    
    # Concatenate the outputs from the branches
    x = concatenate([x, inputs])
    
    # Final 1x1 convolution
    x = Conv2D(filters=3, kernel_size=(1, 1), activation='softmax')(x)  # assuming 3 classes
    
    # Input fusion branch
    input_fusion = Conv2D(filters=32, kernel_size=(1, 1), activation='relu')(inputs)
    x = Add()([x, input_fusion])
    
    # Classification head
    x = GlobalAveragePooling2D()(x)
    outputs = Dense(units=10, activation='softmax')(x)
    
    # Model construction
    model = Model(inputs=inputs, outputs=outputs)
    
    return model

# Build the model
model = dl_model()
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])