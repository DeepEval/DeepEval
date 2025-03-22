import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, BatchNormalization, Flatten, Dense, UpSampling2D

def dl_model():
    # Input layer
    input_layer = Input(shape=(32, 32, 3))
    
    # Main path
    def main_path(input_tensor):
        conv1 = Conv2D(filters=32, kernel_size=(1, 1), padding='same')(input_tensor)
        batch_norm1 = BatchNormalization()(conv1)
        conv_main = Conv2D(filters=32, kernel_size=(3, 3), padding='same')(batch_norm1)
        batch_norm_main = BatchNormalization()(conv_main)
        pool_main = MaxPooling2D(pool_size=(2, 2))(batch_norm_main)
        
        def branch1(input_tensor):
            conv1 = Conv2D(filters=64, kernel_size=(3, 3), padding='same', activation='relu')(input_tensor)
            batch_norm1 = BatchNormalization()(conv1)
            conv2 = Conv2D(filters=64, kernel_size=(3, 3), padding='same', activation='relu')(batch_norm1)
            return conv2
        
        branch1_output = branch1(pool_main)
        
        def branch2(input_tensor):
            pool_branch = MaxPooling2D(pool_size=(2, 2))(input_tensor)
            conv1 = Conv2D(filters=64, kernel_size=(3, 3), padding='same', activation='relu')(pool_branch)
            batch_norm1 = BatchNormalization()(conv1)
            conv2 = Conv2D(filters=64, kernel_size=(3, 3), padding='same', activation='relu')(batch_norm1)
            up_branch = UpSampling2D(size=(2, 2))(conv2)
            return up_branch
        
        branch2_output = branch2(branch1_output)
        
        def branch3(input_tensor):
            pool_branch = MaxPooling2D(pool_size=(2, 2))(input_tensor)
            conv1 = Conv2D(filters=64, kernel_size=(3, 3), padding='same', activation='relu')(pool_branch)
            batch_norm1 = BatchNormalization()(conv1)
            conv2 = Conv2D(filters=64, kernel_size=(3, 3), padding='same', activation='relu')(batch_norm1)
            up_branch = UpSampling2D(size=(2, 2))(conv2)
            return up_branch
        
        branch3_output = branch3(branch1_output)
        
        def final_conv(input_tensor):
            conv1 = Conv2D(filters=3, kernel_size=(1, 1), padding='same')(input_tensor)
            batch_norm1 = BatchNormalization()(conv1)
            flatten = Flatten()(batch_norm1)
            dense1 = Dense(units=128, activation='relu')(flatten)
            dense2 = Dense(units=64, activation='relu')(dense1)
            output = Dense(units=10, activation='softmax')(dense2)
            return output
        
        concat_output = Concatenate()([final_conv(branch2_output), final_conv(branch3_output)])
        output_main = final_conv(concat_output)
        
        model = keras.Model(inputs=input_layer, outputs=output_main)
        return model

    return dl_model()

# Build and compile the model
model = dl_model()
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Model summary
model.summary()