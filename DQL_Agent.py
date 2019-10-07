import numpy as np
import pygame
import cv2
import tensorflow as tf

class DQL(object):
    def __init__(self):
        self.gamma = 0.9
        self.model = self.network()
        self.memory = list()
        self.reward = 0

    def get_state(self, screen):
        arr = pygame.surfarray.array2d(screen)
        arr = np.array(arr).astype('float32')

        return arr

    def network(self):
        model = tf.keras.Sequential([
            tf.keras.layers.Conv2D(32, kernel_size = (3,3), activation = 'relu', input_shape = (28,28,1)),
            tf.keras.layers.Conv2D(64, kernel_size = (3,3), activation = 'relu'),
            tf.keras.layers.MaxPool2D(pool_size = (2,2)),
            tf.keras.layers.Conv2D(64, kernel_size = (3,3), activation = 'relu'),
            tf.keras.layers.Conv2D(128, kernel_size = (3,3), activation = 'relu'),
            tf.keras.layers.MaxPool2D(pool_size = (2,2)),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dropout(0.5),
            tf.keras.layers.Dense(128, activation = 'relu'),
            tf.keras.layers.Dense(3, activation = 'softmax')
        ])

        model.compile(optimizer = 'adam',
                      loss = 'mse',
                      metrics = ['accuracy'])

        return model

    def replay_new(self):
        for state, action, reward, new_state, crash in self.memory:
            target = reward
            if(not crash):
                # state = cv2.cvtColor(state, cv2.COLOR_BGR2GRAY)
                state = cv2.resize(state, (28,28))

                new_state = cv2.resize(new_state, (28,28))

                target = target + self.gamma * np.amax(self.model.predict(np.array([new_state]).reshape(1,28,28,1))[0])
                qValues = self.model.predict(np.array([state]).reshape(1,28,28,1))
                qValues[0][np.argmax(action)] = target

                self.model.fit(np.array([state]).reshape(1,28,28,1), qValues, epochs = 1, verbose = 0)

    def remember(self, tuple):
        self.memory.append(tuple)

    def train_short_memory(self,state, action, reward, new_state, scratch):
        target = reward
        if(not scratch):
            # Preprocess the old_state and new_state first:
            # state = cv2.cvtColor(state, cv2.COLOR_BGR2GRAY)
            state = cv2.resize(state, (28,28))

            new_state = cv2.resize(new_state, (28,28))

            target = target + self.gamma * np.amax(self.model.predict(np.array([new_state]).reshape(1,28,28,1))[0])
            qValues = self.model.predict(np.array([state]).reshape(1,28,28,1))
            qValues[0][np.argmax(action)] = target

            self.model.fit(np.array([state]).reshape(1,28,28,1), qValues, epochs = 1, verbose = 0)
    def set_reward(self, crash, reasonable_shoot):
        self.reward = 0
        if(crash):
            self.reward = -5
        if(reasonable_shoot):
            self.reward = 3
        return self.reward


