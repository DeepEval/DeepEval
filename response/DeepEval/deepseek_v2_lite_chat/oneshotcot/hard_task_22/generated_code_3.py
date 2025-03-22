import keras
from keras.layers import Input, Lambda, Conv2D, MaxPooling2D, Concatenate, BatchNormalization, Flatten, Dense
from keras.models import Model

def dl_model():
    # Input layer
    input_layer = Input(shape=(32, 32, 3))
    
    # Main path
    def main_path(input_tensor):
        # Split input into three groups along the channel
        split1 = Lambda(lambda x: keras.layers.split(x, 3, axis=-1))(input_tensor)
        split2 = Lambda(lambda x: keras.layers.split(x, 3, axis=-1))(input_tensor)
        split3 = Lambda(lambda x: keras.layers.split(x, 3, axis=-1))(input_tensor)
        
        # Multi-scale feature extraction with separable convolutional layers
        conv1_main = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(split1[0])
        conv2_main = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(split1[1])
        conv3_main = Conv2D(filters=64, kernel_size=(5, 5), strides=(1, 1), padding='same', activation='relu')(split1[2])
        
        conv1_branch = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(split2[0])
        conv2_branch = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(split2[1])
        conv3_branch = Conv2D(filters=64, kernel_size=(5, 5), strides=(1, 1), padding='same', activation='relu')(split2[2])
        
        # Concatenate outputs from main path
        concat = Concatenate()(list(concatenate([conv1_main, conv2_main, conv3_main, conv1_branch, conv2_branch, conv3_branch])) if split3 else [conv1_main, conv2_main, conv3_main, conv1_branch, conv2_branch, conv3_branch])
        
        # Batch normalization and flattening
        bn = BatchNormalization()(concat)
        flatten = Flatten()(bn)
        
        # Fully connected layers
        dense1 = Dense(units=128, activation='relu')(flatten)
        dense2 = Dense(units=64, activation='relu')(dense1)
        output_layer = Dense(units=10, activation='softmax')(dense2)
        
        # Model
        model = Model(inputs=input_layer, outputs=output_layer)
        return model
    
    # Branch path
    def branch_path(input_tensor):
        # 1x1 convolutional layer to align channel dimensions
        conv_branch = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        
        # Model
        model_branch = Model(inputs=input_tensor, outputs=conv_branch)
        return model_branch
    
    # Construct the main path and branch path
    model_main = main_path(input_layer)
    model_branch = branch_path(input_layer)
    
    # Combine outputs from main path and branch path
    model = Model(inputs=input_layer, outputs=keras.layers.Add()([model_main.output, model_branch.output]))
    model = Flatten()(model.outputs)
    model = Dense(units=128, activation='relu')(model)
    model = Dense(units=10, activation='softmax')(model)
    
    return model

# Build the model
model = dl_model()
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])