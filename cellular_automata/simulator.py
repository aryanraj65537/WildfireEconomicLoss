import sys, os, time
import pygame
import random
import matplotlib.pyplot as plt
import numpy as np

x = np.arange(20).reshape((4,5))
np.savetxt('map.txt', x)

SC_WIDTH, SC_HEIGHT = 800,800
INITIAL_TREE_DENSITY = 1
MAP_WIDTH = 10
MAP_HEIGHT = 10
TILE_SIZE = 25

trees = []
fires = []
costs = []
listOfBurned = []

pygame.init()

screen = pygame.display.set_mode((SC_WIDTH, SC_HEIGHT))

TREE_IMG = pygame.image.load(os.path.join("Graphics", "Tree_Small.png")).convert_alpha()
FIRE_IMG = pygame.image.load(os.path.join("Graphics", "Fire_Small.png")).convert_alpha()

def createNewForest():
    map = [["T" if random.random()<= INITIAL_TREE_DENSITY else " "
            for x in range(MAP_WIDTH)]
            for y in range(MAP_HEIGHT)]
    map[0][0] = "T"
    return map

def displayForest(forest):
    for x in range(MAP_WIDTH):
        for y in range(MAP_HEIGHT):
            if forest[y][x] == "T":
                screen.blit(TREE_IMG, (x*TILE_SIZE, y*TILE_SIZE))
            elif forest[y][x] == "F":
                screen.blit(FIRE_IMG, (x*TILE_SIZE, y*TILE_SIZE))



def main(SIM_LENGTH, input_for, cburns):
    pygame.init()

    break_out = False
    forest = input_for

    for step in range(SIM_LENGTH):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                break_out = True
        if break_out:
            break      
        tree_count = sum(row.count("T") for row in forest)
        tree_land_percentage = (tree_count/(MAP_HEIGHT*MAP_WIDTH))*100
        trees.append(tree_land_percentage)
        fire_count = sum(row.count("F") for row in forest)
        fire_land_percentage = (fire_count/(MAP_HEIGHT*MAP_WIDTH))*100
        fires.append(fire_land_percentage)

        next_forest = [["Empty" for x in range(MAP_WIDTH)] for y in range(MAP_HEIGHT)]
        #screen.fill((137,234,123))
        #displayForest(forest)
        #pygame.display.update()

        #Build next forest
        forest[0][0] = "F"
        for loc in cburns:
            forest[loc[0]][loc[1]] = " "
        #time.sleep(4)
        for x in range(MAP_WIDTH):
            for y in range(MAP_HEIGHT):
                if next_forest[y][x] != "Empty":
                    continue
                if forest[y][x] == "F":
                    for ix in range(-1,2):
                        for iy in range(-1,2):
                            if (x+ix)>=0 and (y+iy) >= 0:
                                if (x+ix)<=(MAP_WIDTH-1) and (y+iy) <= (MAP_HEIGHT -1):
                                    if forest[y+iy][x+ix] == "T" and (random.random()<= 1):
                                        listOfBurned.append((y+iy, x+ix))
                                        next_forest[y+iy][x+ix] = "F"
                    #delete tree after fire
                    next_forest[y][x] = " "

                else:
                    next_forest[y][x] = forest[y][x]
        forest = next_forest
        
        one_forest = []
        for x in range(len(forest)):
            for y in range(len(forest[x])):
                one_forest.append(forest[x][y])
        if "F" not in one_forest:
            print("no more fire")
            break
            
        time.sleep(0.5)
    #print(listOfBurned)
    
    time.sleep(5)
    pygame.quit()
    return listOfBurned


def get_cost(lob, input_forest):
    cost = 0
    for x in lob:
        cost+=1
    return cost
    

if __name__ == '__main__':
    try:
        main(20, createNewForest(), [(2,1),(1,2), (0,2), (2,0), (2,2)])
    except KeyboardInterrupt:
        sys.exit()