import random
import numpy as np
from collections import deque
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, Flatten, Conv2D, MaxPooling2D
from tensorflow.keras.optimizers import Adam

class DQN:
    def __init__(self, state_size, action_size, gamma, epsilon, update_rate):
        self.state_size = state_size
        self.action_size = action_size
        self.replay_buffer = deque(maxlen=5000)
        self.gamma = gamma
        self.epsilon = epsilon
        self.update_rate = update_rate
        self.main_network = self.build_network()
        self.target_network = self.build_network()
        self.target_network.set_weights(self.main_network.get_weights()) # copy weights of main network to target network
        #self.color = np.array([210, 164, 74]).mean()
        
    #def preprocess_state(self, state):
    #    image = state[1:176:2, ::2] # crop and resize image
    #    image = image.mean(axis=2) # convert image to greyscale
    #    image[image==self.color] = 0 # improve image contrast
    #    image = (image - 128) / 128-1 # normalize image
    #    image = np.expand_dims(image.reshape(88,80,1), axis=0) # reshape image
    #    return image
        
    def build_network(self):
        # define the convolutional layer
        model = Sequential()
        model.add(Conv2D(16, (8,8), strides=4, padding='same', input_shape=self.state_size))
        model.add(Activation('relu'))
        # define second convolutional layer
        model.add(Conv2D(32, (4,4), strides=2, padding='same'))
        model.add(Activation('relu'))
        # define third layer
        model.add(Conv2D(32, (3,3), strides=1, padding='same'))
        model.add(Activation('relu'))
        # flatten the feature maps
        model.add(Flatten())
        # feed the flattened maps to the fully connected layer
        model.add(Dense(256, activation='relu'))
        model.add(Dense(self.action_size, activation='linear'))
        # compile model with mse loss
        model.compile(optimizer=Adam(), loss='mse', metrics=['accuracy'])
        return model
    
    def store_transition(self, state, action, reward, next_state, done):
        self.replay_buffer.append((state, action, reward, next_state, done))
        
    def epsilon_greedy(self, state):
        if random.uniform(0,1) < self.epsilon:
            return np.random.randint(self.action_size)
        Q_values = self.main_network.predict(state)
        return np.argmax(Q_values[0])
    
    def train(self, batch_size
              #, tensorboard_callback
             ):
        # sample a minibatch of transitions from replay buffer
        minibatch = random.sample(self.replay_buffer, batch_size)
        
        # compute target value using target network
        for state, action, reward, next_state, done in minibatch:
            if not done:
                next_state = np.array(next_state).reshape(4,84,84,1)
                target_Q = (reward + self.gamma * np.amax(
                    self.target_network.predict(next_state)))
            else:
                target_Q = reward
        # return predictions from main network and store it in Q_values
        state = np.array(state).reshape(4,84,84,1)
        Q_values = self.main_network.predict(state) 
        # update the target value:
        Q_values[0][action] = target_Q
        # train the main network
        self.main_network.fit(state, Q_values, epochs=1, verbose=0
                             #,callbacks=[tensorboard_callback]
                             )
        
    def update_target_network(self):
        self.target_network.set_weights(self.main_network.get_weights())