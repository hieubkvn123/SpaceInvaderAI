import pygame
import numpy as np
import random
import tensorflow as tf
from os import path
from DQL_bin_state import DQL

img_dir = path.join(path.dirname(__file__), 'img')


class Bullet(pygame.sprite.Sprite):
    def __init__(self,x,y):
        pygame.sprite.Sprite.__init__(self)
        self.image = pygame.Surface((10,20))
        self.image.fill(YELLOW)
        self.rect = self.image.get_rect()
        self.rect.bottom = y
        self.rect.centerx = x
        self.speedy  = -10


    def update(self):
        self.rect.y += self.speedy

        if self.rect.bottom < 0:
            self.kill()

mob_shot = False
class MOB(pygame.sprite.Sprite):
    def __init__(self):
        pygame.sprite.Sprite.__init__(self)
        self.image = pygame.Surface((30,30)) # Mob's height = 30
        self.image.fill(RED)
        self.rect = self.image.get_rect()
        self.rectx = random.randrange(0, WIDTH-self.rect.width)
        self.recty = random.randrange(-100, -40)
        self.speedy = random.randrange(1,8)
        self.speedx = random.randrange(-4,6)

    def update(self):
        self.rect.x += self.speedx
        self.rect.y += self.speedy
        if self.rect.top > HEIGHT+10 or self.rect.left < -25 or self.rect.right > WIDTH +20:
            self.rect.x = random.randrange(WIDTH - self.rect.width)
            self.rect.y = random.randrange(-100,-40)
            self.speedy = random.randrange(1, 8)

class Player(pygame.sprite.Sprite):
    def __init__(self):
        pygame.sprite.Sprite.__init__(self)
        self.image = pygame.Surface((50,50))
        self.image.fill(GREEN)
        self.rect = self.image.get_rect()
        self.rect.centerx = WIDTH/2
        self.rect.bottom = HEIGHT - 10
        self.speedx = 0
        self.mov_sped = 10

    def update(self):
        self.speedx = 0

        if(ai_mode):
            self.move(predicted_move)
        else:
            self.controls()

        self.constraint()

    def move(self, move):
        if (move == 0):  # moving right
            self.speedx = player.mov_sped
            self.rect.x += self.speedx
        if (move == 1):  # moving left
            self.speedx = -player.mov_sped
            self.rect.x += self.speedx
        if (move == 2):
            self.shoot()
    def controls(self):
        keystate = pygame.key.get_pressed()
        if keystate[pygame.K_LEFT] or keystate[pygame.K_a]:
            self.speedx = -self.mov_sped
        if keystate[pygame.K_RIGHT] or keystate[pygame.K_d]:
            self.speedx = self.mov_sped

        self.rect.x += self.speedx

        #if keystate[pygame.K_SPACE]:
            #self.shoot()

    def constraint(self):
        if self.rect.right > WIDTH:
            self.rect.right = WIDTH
        if self.rect.left < 0:
            self.rect.left = 0

    def shoot(self):
        bullet = Bullet(self.rect.centerx, self.rect.top)
        all_sprites.add(bullet)
        bullets.add(bullet)

def generate_pos():
    for i in range(50):
        x = random.randrange(0, WIDTH)
        y = random.randrange(0, HEIGHT)
        star_list.append([x,y])

def generate_background(star_list):

    for i in range(len(star_list)):

        pygame.draw.circle(screen, WHITE, star_list[i],2)

        star_list[i][1] += 1

        if star_list[i][1] > HEIGHT:

            y = random.randrange(-50,-10)
            star_list[i][1] = y
            x = random.randrange(0, WIDTH)
            star_list[i][0] = x

##############################

#Colors
BLACK =(0,0,0)
WHITE =(255,255,255)
RED   =(255,0,0)
GREEN =(0,255,0)
BLUE  =(0,0,255)
YELLOW = (255,255,0)

WIDTH = 480
HEIGHT = 600
##############################

##init.py
pygame.init()

size=(WIDTH,HEIGHT)
screen = pygame.display.set_mode(size)

pygame.display.set_caption("Space Destroyer")
running = True
clock = pygame.time.Clock()

#background = pygame.image.load(path.join(img_dir, "blue.png")).convert()
#background_rect  = background.get_rect()
all_sprites = pygame.sprite.Group()
mobs = pygame.sprite.Group()
bullets = pygame.sprite.Group()
player = Player()
all_sprites.add(player)

for i in range(8):
    m = MOB()
    all_sprites.add(m)
    mobs.add(m)

############################

star_list = []
generate_pos()
keypress = pygame.key.get_pressed()

agent = DQL()

#-------MAIN PROGRAM LOOP--------#
# actions will be stored in a list like [right, left, shoot]
ai_mode = True
predicted_move = None

if(not ai_mode):
    while  running:
        mob_shot = False # reset mob shot
        reasonable_shoot = False
        crash = False
        state = agent.get_state(player,mobs)
        reward = 0

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE:
                    player.shoot()
                    if (state[5] == 1): reasonable_shoot = True

        hits = pygame.sprite.groupcollide(mobs,bullets, True, True)
        if(len(hits) > 0): mob_shot = True
        for hit in hits:
            m = MOB()
            all_sprites.add(m)
            mobs.add(m)

        hits = pygame.sprite.spritecollide(player, mobs, False)
        if hits:
            running  = False
            crash = True

        reward = agent.set_reward(crash, reasonable_shoot)
        if(reward > 0 ): print(reward)

        screen.fill(BLACK)

        generate_background(star_list)

        all_sprites.update()
        all_sprites.draw(screen)

        pygame.display.flip()
        clock.tick(60)

    print(reward)
    pygame.quit()
else :
    game_count = 0
    crash = False
    epsilon = 80
    while (game_count < 150):
        game_count += 1
        epsilon -= 1 # epsilon is decayed as the model is well trained
        print("Game " + str(game_count))
        crash = False

        # Initialize the game once again
        all_sprites = pygame.sprite.Group()
        mobs = pygame.sprite.Group()
        bullets = pygame.sprite.Group()
        player = Player()
        all_sprites.add(player)

        for i in range(8):
            m = MOB()
            all_sprites.add(m)
            mobs.add(m)

        star_list = []
        generate_pos()

        agent.replay_new()
        while not crash:
            mob_shot = False  # reset mob shot
            reasonable_shoot = False
            crash = False
            reward = 0

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
            # get old state here
            state = agent.get_state(player, mobs)

            # execute actions here; action a = [right, left, shoot]
            if(np.random.randint(0,100) < epsilon ): # do a random move
                predicted_move = np.random.randint(0,3)
                action = tf.keras.utils.to_categorical(predicted_move, num_classes = 3)
            else:
                action = agent.model.predict(np.array([state]))[0]
                predicted_move = np.argmax(action)

            print(action)
            hits = pygame.sprite.groupcollide(mobs, bullets, True, True)
            for hit in hits:
                m = MOB()
                all_sprites.add(m)
                mobs.add(m)

            hits = pygame.sprite.spritecollide(player, mobs, False)
            if hits:
                crash = True
                break

            # Reward here
            reward = agent.set_reward(crash, reasonable_shoot)

            screen.fill(BLACK)

            generate_background(star_list)

            all_sprites.update()
            all_sprites.draw(screen)
            # GET new state here
            new_state = agent.get_state(player, mobs)

            agent.remember((state, action, reward, new_state, crash))
            agent.train_short_memory(state, action, reward, new_state, crash)

            pygame.display.flip()
            clock.tick(60)

    print(reward)
    pygame.quit()