import socketio
import random
from .gen_api import *
from typing import List, Tuple, Union, Dict, Any
from collections import deque


class Point:
    def __init__(self, x: int, y: int):
        self.x = x
        self.y = y

class GBot(GBotBase):
    def __init__(self, room_id: str, username: str = "GenniaBot"):
        super().__init__(room_id,username)
        self.color = None
        self.init_game_info = None
        self.game_map = None
        self.turns_count = 0  # 新增回合计数器
        self.enemy_visable=False
        self.distab=None

    
    def bfs(self, start: List[Tuple[int,int]], max_distance=9999) -> Dict[Tuple[int,int], int]:
        """广度优先搜索计算距离"""
        distances = {}
        queue = deque()
        for s in start:
            queue.append(s)
            distances[s] = 0
        directions = [(0, 1), (0, -1), (1, 0), (-1, 0)]
        map_width = len(self.game_map)
        map_height = len(self.game_map[0])
        while queue:
            #print(len(queue),len(distances))
            current = queue.popleft()
            current_dist = distances[current]
            if current_dist >= max_distance:
                continue
            for d in directions:    
                nb=(current[0]+d[0],current[1]+d[1])
                if nb[0]<0 or nb[0]>=map_width or nb[1]<0 or nb[1]>=map_height:
                    continue
                if self.game_map[nb[0]][nb[1]].color_index==self.color or self.game_map[nb[0]][nb[1]].tile_type==TileType.Fog or self.game_map[nb[0]][nb[1]].tile_type==TileType.Plain:    
                    if nb not in distances:
                        distances[nb] = current_dist + 1
                        queue.append(nb)
        #print("bfs end")            
        return distances
    def patch_map(self, map_diff: List[Union[int, TilePropTuple]]):
        if not self.game_map:
            return
        map_width = len(self.game_map)
        map_height = len(self.game_map[0])
        flattened = [tile for row in self.game_map for tile in row]
        new_state = [[None for _ in range(map_height)] for _ in range(map_width)]
        i = j = 0
        for diff in map_diff:
            if isinstance(diff, int):
                j += diff
            else:
                flattened[j] = TileProp(*diff)
                j += 1
        for i in range(map_width):
            for j in range(map_height):
                new_state[i][j] = flattened[i * map_height + j]
        self.game_map = new_state
        self.enemy_visable=False
        vise=[]
        for i in range(map_width):
            for j in range(map_height):
                if self.game_map[i][j].color_index!=self.color and self.game_map[i][j].color_index:
                    self.enemy_visable=True
                    vise.append((i,j))
        self.distab=self.bfs(vise)

    def evaluate_move(self, source: Point, direction: Tuple[int, int]) -> float:
        nx, ny = source.x + direction[0], source.y + direction[1]
        # 边界检查
        if nx < 0 or ny < 0 or nx >= len(self.game_map) or ny >= len(self.game_map[0]):
            return -1
        
        target_tile = self.game_map[nx][ny]
        source_tile = self.game_map[source.x][source.y]
        move_army = source_tile.army_size - 1  # 可移动兵力

        # 1. 目标为迷雾（探索）
        if target_tile.tile_type == TileType.Fog:
            return 10 if self.turns_count < 25 else 5
        
        # 2. 目标为山地
        if target_tile.tile_type == TileType.Mountain:  # 山地
            return -1
        
        df=(move_army-target_tile.army_size)

        # 3. 目标为中立单位（空地/要塞）
        if not target_tile.color_index or target_tile.color_index == 0:
            if target_tile.tile_type == TileType.City:  # 中立要塞
                if self.enemy_visable:
                    return 15+df/6 if move_army >= target_tile.army_size + 2 else 0
                else:
                    return 20+df/6 if move_army >= target_tile.army_size + 2 else 0
            return 15  # 空地
        
        # 4. 目标为敌方单位
        if target_tile.color_index != self.color:
            if target_tile.tile_type == TileType.King:  # 敌方首都
                return 1000 if move_army >= target_tile.army_size + 2 else -5
            elif move_army >= target_tile.army_size + 2:  # 可占领
                if target_tile.tile_type==TileType.City:
                    return 25+(move_army-target_tile.army_size)/4
                else:
                    return 25+(move_army-target_tile.army_size)/6
            elif move_army >= target_tile.army_size:  # 消耗战
                return 5
            return -5  # 兵力不足
        
        # 5. 目标为己方单位（集结）
        score = 10+((target_tile.army_size-1)/8 if source_tile.army_size>target_tile.army_size else 0) if self.turns_count >= 50 else 3
        
        # 首都保护：前期减少移动首都兵力
        if source_tile.tile_type == TileType.King:
            score *= 0.2 if self.turns_count < 25 else 0.8
        ma_scalar=0.12
        if self.enemy_visable and (nx,ny) in self.distab and self.turns_count>100 and move_army>self.turns_count:
            sd=self.distab[(source.x,source.y)]
            td=self.distab[(nx,ny)]
            if td<sd:
                ma_scalar=0.15
        score += move_army * ma_scalar
        return score

    def handle_move(self):
        if not self.game_map or not self.init_game_info or not self.color:
            return
        
        # 收集所有可移动格子（兵力>1）
        lands = []
        for i in range(len(self.game_map)):
            for j in range(len(self.game_map[0])):
                tile = self.game_map[i][j]
                if tile.color_index == self.color and tile.army_size > 1:
                    lands.append(Point(i, j))
        
        # 评估所有可能移动
        moves = []
        directions = [(0, 1), (0, -1), (1, 0), (-1, 0)]
        for source in lands:
            for direction in directions:
                score = self.evaluate_move(source, direction)
                if score < 0:  # 跳过无效移动
                    continue
                moves.append((source, direction, score))
        
        # 选择最佳移动
        if not moves:
            return
        
        best_moves = []
        max_score = max(moves, key=lambda x: x[2])[2]
        for move in moves:
            if move[2] == max_score:
                best_moves.append(move)
        
        source, direction, _ = random.choice(best_moves)
        target_point = {"x": source.x + direction[0], "y": source.y + direction[1]}
        
        return ({"x": source.x, "y": source.y}, target_point,False)
