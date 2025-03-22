import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Dropout, Conv2DTranspose, Add, concatenate, BatchNormalization, LeakyReLU, Activation

def dl_model():
    # Input layer
    input_layer = Input(shape=(32, 32, 3))
    
    # First part: feature extraction
    def extract_features(input_tensor):
        # Pair 1: Conv -> MaxPool
        conv1 = Conv2D(filters=32, kernel_size=(3, 3), activation='relu')(input_tensor)
        pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
        
        # Pair 2: Conv -> MaxPool
        conv2 = Conv2D(filters=64, kernel_size=(3, 3), activation='relu')(pool1)
        pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
        
        # Pair 3: Conv -> MaxPool
        conv3 = Conv2D(filters=128, kernel_size=(3, 3), activation='relu')(pool2)
        pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
        
        return conv3, pool3
    
    # Second part: feature enhancement
    def enhance_features(input_tensor):
        # Dropout layer
        drop1 = Dropout(0.5)(input_tensor)
        
        # Second Conv layer
        conv4 = Conv2D(filters=128, kernel_size=(1, 1), activation='relu')(drop1)
        
        # Second MaxPool layer
        pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)
        
        # Third Conv layer
        conv5 = Conv2D(filters=64, kernel_size=(1, 1), activation='relu')(pool4)
        
        # Second MaxPool layer
        pool5 = MaxPooling2D(pool_size=(2, 2))(conv5)
        
        return conv5, pool5
    
    # Third part: feature upsampling and classification
    def upsample_features(input_tensor):
        # Transposed Conv layer
        trans1 = Conv2DTranspose(filters=64, kernel_size=(2, 2), strides=(2, 2), padding='same')(input_tensor)
        
        # Skip connection
        concat1 = Add()([input_tensor, trans1])
        
        # Batch normalization
        batch_norm1 = BatchNormalization()(concat1)
        
        # Leaky ReLU activation
        leaky_relu1 = LeakyReLU(alpha=0.1)(batch_norm1)
        
        # Second Transposed Conv layer
        trans2 = Conv2DTranspose(filters=32, kernel_size=(2, 2), strides=(2, 2), padding='same')(leaky_relu1)
        
        # Skip connection
        concat2 = Add()([concat1, trans2])
        
        # Batch normalization
        batch_norm2 = BatchNormalization()(concat2)
        
        # Leaky ReLU activation
        leaky_relu2 = LeakyReLU(alpha=0.1)(batch_norm2)
        
        # Second Transposed Conv layer
        trans3 = Conv2DTranspose(filters=16, kernel_size=(2, 2), strides=(2, 2), padding='same')(leaky_relu2)
        
        # Skip connection
        concat3 = Add()([concat2, trans3])
        
        # Batch normalization
        batch_norm3 = BatchNormalization()(concat3)
        
        # Leaky ReLU activation
        leaky_relu3 = LeakyReLU(alpha=0.1)(batch_norm3)
        
        # Output layer
        output_layer = Conv2D(filters=10, kernel_size=(1, 1), activation='softmax')(leaky_relu3)
        
        return output_layer
    
    # Construct the model
    conv3, pool3 = extract_features(input_layer)
    conv5, pool5 = enhance_features(conv3)
    output_layer = upsample_features(conv5)
    
    model = keras.Model(inputs=input_layer, outputs=output_layer)
    
    return model

model = dl_model()
model.summary()