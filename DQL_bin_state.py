import numpy as np
import pandas as pd
import pygame
import cv2
import tensorflow as tf

# action list will be [right, left, shoot]

class DQL(object): # not done on overall
    def __init__(self):
        self.gamma = 0.9
        self.model = self.network()
        self.memory = list()

    def get_state(self, player, mobs_group): # done
        danger_range = 100
        mob_size = 30
        player_center = player.rect.x + 25
        vision_range = 50

        left_bound = player_center - vision_range
        right_bound = player_center + vision_range
        if(isinstance(player, object) and isinstance(mobs_group, pygame.sprite.Group)):
            edge_left = 0
            edge_right = 0
            danger_left = 0
            danger_top = 0
            danger_right = 0
            mobs_ahead = 0

            # Edge left
            if(player.rect.x == 0): edge_left = 1
            if(player.rect.x + 50 == 480): edge_right = 1

            danger_top_corner = [player.rect.x -25, player.rect.y - 100]
            danger_left_corner = [player.rect.x - 100, player.rect.y - 50]
            danger_right_corner = [player.rect.x + 50, player.rect.y - 50]

            for mob in mobs_group.sprites():
                mob_bottom_left = [mob.rect.x, mob.rect.y+mob_size]
                mob_bottom_right = [mob.rect.x+mob_size, mob.rect.y + mob_size]

                if(((mob_bottom_left[0] in range(danger_left_corner[0], danger_left_corner[0] + danger_range)) and (mob_bottom_left[1] in range(danger_left_corner[1], danger_left_corner[1] + danger_range)))\
                or (mob_bottom_right[0] in range(danger_left_corner[0], danger_left_corner[0] + danger_range)) and (mob_bottom_right[1] in range(danger_left_corner[1], danger_left_corner[1] + danger_range))):
                    danger_left = 1

                if (((mob_bottom_left[0] in range(danger_right_corner[0], danger_right_corner[0] + danger_range)) and (mob_bottom_left[1] in range(danger_right_corner[1], danger_right_corner[1] + danger_range))) \
                or (mob_bottom_right[0] in range(danger_right_corner[0],danger_right_corner[0] + danger_range)) and (mob_bottom_right[1] in range(danger_right_corner[1],danger_right_corner[1] + danger_range))):
                    danger_right = 1

                if (((mob_bottom_left[0] in range(danger_top_corner[0], danger_top_corner[0] + danger_range)) and (mob_bottom_left[1] in range(danger_top_corner[1], danger_top_corner[1] + danger_range))) \
                or (mob_bottom_right[0] in range(danger_top_corner[0],danger_top_corner[0] + danger_range)) and (mob_bottom_right[1] in range(danger_top_corner[1],danger_top_corner[1] + danger_range))):
                    danger_top = 1

                if(mob.speedx > 0 and (mob_bottom_left[0] in range(left_bound, player_center) or mob_bottom_right[0] in range(left_bound, player_center))):
                    mobs_ahead = 1
                if(mob.speedx < 0 and (mob_bottom_left[0] in range(player_center, right_bound) or mob_bottom_right[0] in range(player_center, right_bound))):
                    mobs_ahead = 1

        return [edge_left, edge_right, danger_left, danger_top, danger_right, mobs_ahead]


    def network(self): # done
        model = tf.keras.Sequential([
            tf.keras.layers.Dense(6, activation = 'relu', input_shape = (6,)),
            tf.keras.layers.Dense(8, activation = 'relu'),
            tf.keras.layers.Dense(8, activation = 'relu'),
            tf.keras.layers.Dense(3, activation = 'softmax')
        ])

        model.compile(optimizer = 'adam',
                      loss = 'mse', # if you want to train in a way that from vector -> vector, gotta use mse as loss function
                                    # if it's just a simple classification task, sparse_categorical_crossentropy is fine
                      metrics = ['accuracy'])

        return model

    def replay_new(self):
        for state, action, reward, new_state, scratch in self.memory:
            target = reward
            if (not scratch):
                target = target + self.gamma * np.max(self.model.predict(np.array([new_state]))[0])
            action[np.argmax(action)] = target

            self.model.fit(np.array([state]), np.array([action]), epochs=1, verbose=0)

    def remember(self, tuple): # done
        self.memory.append(tuple)

    def train_short_memory(self,state, action, reward, new_state, scratch): # not done
        target = reward
        if(not scratch):
            target = target + self.gamma * np.max(self.model.predict(np.array([new_state]))[0])
        qValues = self.model.predict(np.array([state]))
        qValues[0][np.argmax(action)] = target

        print(np.array([state]), qValues)
        self.model.fit(np.array(state).reshape(1,6), qValues, epochs = 1, verbose = 0)

    def set_reward(self, crash, reasonable_shoot): # done
        reward = 0
        if(crash): reward = -5
        if(reasonable_shoot): reward = 2

        return reward
