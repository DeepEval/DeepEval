import keras
from keras.models import Model
from keras.layers import Input, Conv2D, MaxPooling2D, Dropout, Add, AveragePooling2D, Flatten, Dense

def dl_model():
    # Input layer
    input_layer = Input(shape=(28, 28, 1))
    
    # Main pathway
    def main_path(x):
        conv1 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(x)
        conv2 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(conv1)
        pool1 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid')(conv2)
        dropout1 = Dropout(0.5)(pool1)
        return dropout1
    
    # Branch pathway
    def branch_path(x):
        conv = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(x)
        pool = AveragePooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(conv)
        return pool
    
    # Main pathway processing
    main_output = main_path(input_layer)
    
    # Branch pathway processing
    branch_output = branch_path(input_layer)
    
    # Fusion of main and branch pathways
    fused_output = Add()([main_output, branch_output])
    
    # Global average pooling and fully connected layer
    pool1 = AveragePooling2D(pool_size=(7, 7), strides=(1, 1), padding='valid')(fused_output)
    flatten = Flatten()(pool1)
    dense = Dense(units=128, activation='relu')(flatten)
    output = Dense(units=10, activation='softmax')(dense)
    
    # Construct the model
    model = Model(inputs=input_layer, outputs=output)
    
    return model

# Build the model
model = dl_model()
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])