from tensorflow.keras.applications import VGG16
from tensorflow.keras.layers import Input, Conv2D, BatchNormalization, ReLU, Add, GlobalAveragePooling2D, Dense
from tensorflow.keras.models import Model

def dl_model():
    # Input layer
    inputs = Input(shape=(32, 32, 3))
    
    # Load a pretrained model as a feature extractor
    base_model = VGG16(weights='imagenet', include_top=False, input_shape=(32, 32, 3))
    base_model.trainable = False  # Freeze the base model
    
    # First block
    x = Conv2D(64, (3, 3), activation='relu')(inputs)
    x = Conv2D(64, (3, 3), activation='relu')(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    
    # Second block
    x = Conv2D(128, (3, 3), activation='relu')(x)
    x = Conv2D(128, (3, 3), activation='relu')(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    
    # Third block
    x = Conv2D(256, (3, 3), activation='relu')(x)
    x = Conv2D(256, (3, 3), activation='relu')(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    
    # Parallel branch
    x = Conv2D(64, (1, 1), activation='relu')(inputs)
    
    # Concatenate outputs of sequential blocks and parallel branch
    concat = Add()([x, inputs])
    
    # Flatten and pass through fully connected layers
    x = GlobalAveragePooling2D()(concat)
    outputs = Dense(10, activation='softmax')(x)
    
    # Model
    model = Model(inputs=inputs, outputs=outputs)
    
    return model

# Create the model
model = dl_model()
model.summary()