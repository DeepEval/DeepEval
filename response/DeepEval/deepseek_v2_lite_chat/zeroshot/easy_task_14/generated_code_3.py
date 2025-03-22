from keras.models import Model
from keras.layers import Input, Conv2D, GlobalAveragePooling2D, Dense, Flatten
from keras.applications.vgg16 import VGG16

# Load the pre-trained VGG16 model without the last few layers
base_model = VGG16(weights='imagenet', include_top=False, input_shape=(32, 32, 3))

def dl_model():
    # Input layer
    inputs = Input(shape=(32, 32, 3))
    
    # Base model without top layers
    base = base_model(inputs)
    
    # Global average pooling
    avg_layer = GlobalAveragePooling2D()(base)
    
    # Fully connected layers
    dense1 = Dense(1024, activation='relu')(avg_layer)
    dense2 = Dense(1024, activation='relu')(dense1)
    
    # Output layer
    output = Dense(10, activation='softmax')(dense2)
    
    # Create the model
    model = Model(inputs=inputs, outputs=output)
    
    return model

# Create and compile the model
model = dl_model()
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

return model