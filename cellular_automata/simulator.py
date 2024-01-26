import sys, os, time
import pygame
import random
import matplotlib.pyplot as plt
import numpy as np
import networkx as nx

x = np.arange(20).reshape((4,5))
np.savetxt('map.txt', x)

SC_WIDTH, SC_HEIGHT = 4000,4000
INITIAL_TREE_DENSITY = 1
MAP_WIDTH = 1000
MAP_HEIGHT = 1000
TILE_SIZE = 16

trees = []
fires = []
costs = []
listOfBurned = []

pygame.init()

screen = pygame.display.set_mode((SC_WIDTH, SC_HEIGHT))

TREELOW_IMG = pygame.image.load(os.path.join("cellular_automata", "Graphics", "low-biomass.png")).convert_alpha()
TREEMED_IMG = pygame.image.load(os.path.join("cellular_automata", "Graphics", "med-biomass.png")).convert_alpha()
TREEHIGH_IMG = pygame.image.load(os.path.join("cellular_automata", "Graphics", "high-biomass.png")).convert_alpha()
FIRE_IMG = pygame.image.load(os.path.join("cellular_automata", "Graphics", "Fire-Small.png")).convert_alpha()

def createNewForest():
    map = [[random.choice(["L", "M", "H", "F"]) if random.random()<= INITIAL_TREE_DENSITY else " "
            for x in range(MAP_WIDTH)]
            for y in range(MAP_HEIGHT)]
    return map

def displayForest(forest):
    for x in range(MAP_WIDTH):
        for y in range(MAP_HEIGHT):
            if forest[y][x] == "L":
                screen.blit(TREELOW_IMG, (x*TILE_SIZE, y*TILE_SIZE))
            elif forest[y][x] == "F":
                screen.blit(FIRE_IMG, (x*TILE_SIZE, y*TILE_SIZE))
            elif forest[y][x] == "M":
                screen.blit(TREEMED_IMG, (x*TILE_SIZE, y*TILE_SIZE))
            elif forest[y][x] == "H":
                screen.blit(TREEHIGH_IMG, (x*TILE_SIZE, y*TILE_SIZE))


def main(SIM_LENGTH, input_for, cburns):
    pygame.init()

    break_out = False
    forest = input_for

    treeOfBurned = nx.Graph()

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
        screen.fill((137,234,123))
        displayForest(forest)
        pygame.display.update()

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
                                        treeOfBurned.add_edge((y, x), (y+iy, x+ix))
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
    return [listOfBurned, treeOfBurned]


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