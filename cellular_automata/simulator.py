import sys, os, time
import pygame
import random
import matplotlib.pyplot as plt
import numpy as np
import networkx as nx

x = np.arange(20).reshape((4,5))
np.savetxt('map.txt', x)

SC_WIDTH, SC_HEIGHT = 1200,1200
INITIAL_TREE_DENSITY = 1
MAP_WIDTH = 50
MAP_HEIGHT = 50
TILE_SIZE = 15

trees = []
fires = []
costs = []
setOfBurned = set()

pygame.init()

screen = pygame.display.set_mode((SC_WIDTH, SC_HEIGHT))

#graphics
TREELOW_IMG = pygame.image.load(os.path.join("cellular_automata", "Graphics", "low-biomass.png")).convert_alpha()
TREEMED_IMG = pygame.image.load(os.path.join("cellular_automata", "Graphics", "med-biomass.png")).convert_alpha()
TREEHIGH_IMG = pygame.image.load(os.path.join("cellular_automata", "Graphics", "high-biomass.png")).convert_alpha()
FIRE_IMG = pygame.image.load(os.path.join("cellular_automata", "Graphics", "Fire-Small.png")).convert_alpha()
BURNT = pygame.image.load(os.path.join("cellular_automata", "Graphics", "blackimgfinal.png")).convert_alpha()


def createNewForest():
    map = [[random.choice(["L", "M", "H"]) if random.random()<= INITIAL_TREE_DENSITY else " "
            for x in range(MAP_WIDTH)]
            for y in range(MAP_HEIGHT)]
    map[MAP_WIDTH//2][MAP_HEIGHT//2] = "F"
    costList = [[1, 45, 78, 23, 11], [64, 89, 37, 5, 92], [14, 68, 31, 57, 9], [76, 20, 83, 50, 3], [96, 42, 74, 28, 60]]
    mapCosts = costList
    #mapCosts = [[random.randint(1,100)
    #        for x in range(MAP_WIDTH)]
    #        for y in range(MAP_HEIGHT)]

    #1253   15178.941804707934  0.9599123001098633
    
    mapEl = [[831
            for x in range(MAP_WIDTH)]
            for y in range(MAP_HEIGHT)]
    #14424, 14869
    mapTemp = [[14699
            for x in range(MAP_WIDTH)]
            for y in range(MAP_HEIGHT)]
    #1,156
    mapBio = [[75
            for x in range(MAP_WIDTH)]
            
            for y in range(MAP_HEIGHT)]
    '''
    mapCosts = [[random.randint(1,100)
            for x in range(MAP_WIDTH)]
            for y in range(MAP_HEIGHT)]
    mapEl = [[random.randint(1250,1300)
            for x in range(MAP_WIDTH)]
            for y in range(MAP_HEIGHT)]
    #14424, 14869
    mapTemp = [[random.randint(15174,15208)
            for x in range(MAP_WIDTH)]
            for y in range(MAP_HEIGHT)]
    #1,156
    mapBio = [[random.uniform(16,17)
            for x in range(MAP_WIDTH)]
            for y in range(MAP_HEIGHT)]
    '''
    return map, mapCosts, mapEl, mapTemp, mapBio

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
            elif forest[y][x] == "BURNT":
                screen.blit(BURNT, (x*TILE_SIZE, y*TILE_SIZE))

def main(SIM_LENGTH, input_for, cburns, rawFor):
    pygame.init()

    break_out = False

    forest = [x[:] for x in rawFor]
    forCost = input_for[1]
    forEl = input_for[2]
    forTemp = input_for[3]
    forBio = input_for[4]

    renamings = dict()

    renamings[(MAP_WIDTH//2, MAP_HEIGHT//2)] = 0

    lastunusedrenaming = 1

    treeOfBurned = nx.Graph()
    normalizedtree = nx.Graph()
    #print(rawFor)
    for step in range(SIM_LENGTH):
        '''
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                break_out = True
        if break_out:
            break      
        '''
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
        for loc in cburns:
            forest[loc[0]][loc[1]] = "BURNT"

        for x in range(MAP_WIDTH):
            for y in range(MAP_HEIGHT):
                if next_forest[y][x] != "Empty":
                    continue
                if forest[y][x] == "F":
                    for ix in range(-1,2):
                        for iy in range(-1,2):
                            if (x+ix)>=0 and (y+iy) >= 0:
                                if (x+ix)<=(MAP_WIDTH-1) and (y+iy) <= (MAP_HEIGHT -1):
                                    tOF = (random.random() < 0.33)
                                    if (forest[y+iy][x+ix] == "L" or forest[y+iy][x+ix] == "M" or forest[y+iy][x+ix] == "H") and (tOF):
                                        if (y+iy, x+ix) not in setOfBurned:
                                            treeOfBurned.add_edge((y, x), (y+iy, x+ix))
                                            normalizedtree.add_edge(renamings[(y, x)], lastunusedrenaming)
                                            renamings[(y+iy, x+ix)] = lastunusedrenaming
                                            lastunusedrenaming += 1
                                        setOfBurned.add((y+iy, x+ix))
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
            #print("no more fire")
            break
            
        time.sleep(0.5)
    time.sleep(3)
    pygame.quit()

    #totalCost = 0
    #for x in setOfBurned:
        #totalCost += forCost[x[0]][x[1]]
    #print(totalCost)
    return [setOfBurned, treeOfBurned, normalizedtree, renamings]

