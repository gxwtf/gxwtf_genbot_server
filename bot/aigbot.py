from .gen_api import *
import random
from . import ai_model
import numpy as np
import math
from typing import Dict
from collections import deque

class Point:
    def __init__(self, x: int, y: int):
        self.x = x
        self.y = y
    def __eq__(self, other):
        return self.x == other.x and self.y == other.y
ORIGINAL_MAP_WIDTH=23
MAP_CHANNELS=11
def pad(state, fill_value = 0, map_width = ORIGINAL_MAP_WIDTH):
    x_diff = float(map_width - state.shape[1]) / 2
    x_padding = (math.ceil(x_diff), math.floor(x_diff))
    y_diff = float(map_width - state.shape[0]) / 2
    y_padding = (math.ceil(y_diff), math.floor(y_diff))
    #print("hhhh --", y_padding, x_padding, state.shape)
    return np.pad(state, pad_width=(y_padding, x_padding), 
                mode='constant', constant_values=fill_value), y_padding, x_padding

def dtp2dir(direction: Tuple[int, int])->int:
    if direction[0]==-1:
        return 0
    if direction[1]==1:
        return 1
    if direction[1]==-1:
        return 3
    if direction[0]==1:
        return 2

def zero2none(x):
    if x==0:
        return None
    return x

class GBot(GBotBase):
    def __init__(self, room_id: str, username: str = "GenniaBot"):
        super().__init__(room_id,username)
        self.color = None
        self.init_game_info = None
        self.game_map = None
        self.turns_count = 0  # 新增回合计数器
        self.enemy_visable=False
        self.distab=None
        self.game_state=np.zeros((ORIGINAL_MAP_WIDTH, ORIGINAL_MAP_WIDTH, MAP_CHANNELS)).astype('float32')
        self.model=ai_model.Model(ckpt_dir="./epoch2.pt")
        self.tile_pos=None
        self.move_dir=None
        self.rep_pen=None
        self.king_position = None
        self.defense_mode = False
        self.collect_time=0
        self.game_mode=0
        self.x_offset=0
        self.y_offset=0

    def init_map(self, map_width: int, map_height: int):
        self.game_map = [
            [TileProp(TileType.Fog, None, None) for _ in range(map_height)]
            for _ in range(map_width)
        ]
        self.game_state=np.zeros((ORIGINAL_MAP_WIDTH, ORIGINAL_MAP_WIDTH, MAP_CHANNELS)).astype('float32')
        self.rep_pen=np.zeros((map_width,map_height,4))
        self.king_position=None
        self.defense_mode = False
        self.collect_time=0
        self.game_mode=0
        
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
    
    def find_king_position(self):
        """查找国王位置"""
        if self.game_map and self.color:
            for i in range(len(self.game_map)):
                for j in range(len(self.game_map[0])):
                    tile = self.game_map[i][j]
                    if tile.tile_type == TileType.King and tile.color_index == self.color:
                        self.king_position = Point(i, j)
                        return self.king_position
        return None
    
    def check_king_threat(self):
        """检查国王威胁级别"""
        # 每5回合检查一次
        if self.turns_count%8!=0 and self.defense_mode:
            return
            
        # 确保国王位置已找到
        if not self.king_position:
            self.find_king_position()
            if not self.king_position:
                return
        self.defense_mode=False
        map_height=len(self.game_map)
        king_army=self.game_map[self.king_position.x][self.king_position.y].army_size        
        
        # 检查国王周围5格内是否有敌方单位
        for dx in range(-8, 9):
            for dy in range(-8, 9):
                    
                nx, ny = self.king_position.x + dx, self.king_position.y + dy
                if 0 <= nx < len(self.game_map) and 0 <= ny < len(self.game_map[0]):
                    tile = self.game_map[nx][ny]
                    if tile.color_index != self.color and tile.color_index is not None and tile.army_size is not None:
                        # 计算威胁级别：敌方兵力 + 距离权重
                        distance = abs(dx) + abs(dy)
                        if tile.army_size-2*distance>=king_army:
                            self.defense_mode=True
        
    def find_defense_move(self):
        """寻找防御移动 - 保护国王"""
        if not self.defense_mode or not self.king_position:
            return None
            
        # 1. 寻找国王周围的己方单位
        defense_units = []
        for dx in range(-8, 9):
            for dy in range(-8, 9):
                if dx==0 and dy==0:
                    continue
                nx, ny = self.king_position.x + dx, self.king_position.y + dy
                if 0 <= nx < len(self.game_map) and 0 <= ny < len(self.game_map[0]):
                    tile = self.game_map[nx][ny]
                    if tile.color_index == self.color and tile.army_size > 1:
                        defense_units.append(Point(nx, ny))
        
        if not defense_units:
            return None
            
        # 2. 寻找最近的威胁
        nearest_threat = None
        min_distance = float('inf')
        map_height=len(self.game_map)
        for dx in range(-map_height//2, map_height//2+1):
            for dy in range(-map_height//2, map_height//2+1):
                nx, ny = self.king_position.x + dx, self.king_position.y + dy
                if 0 <= nx < len(self.game_map) and 0 <= ny < len(self.game_map[0]):
                    tile = self.game_map[nx][ny]
                    if tile.color_index != self.color and tile.color_index is not None:
                        distance = abs(dx) + abs(dy)
                        if distance < min_distance:
                            min_distance = distance
                            nearest_threat = Point(nx, ny)
        
        best=None
        if min_distance<=8:
            distable=self.bfs([(self.king_position.x,self.king_position.y),(nearest_threat.x,nearest_threat.y)])
        else:
            distable=self.bfs([(nearest_threat.x,nearest_threat.y)])
        # 3. 寻找可以拦截威胁的防御单位
        for source in defense_units:
            # 计算到威胁的路径
            #score=self.game_map[source.x][source.y].army_size*(16-abs(source.x-self.king_position.x)-abs(source.y-self.king_position.y))
            score=self.game_map[source.x][source.y].army_size*(map_height-distable[source.x,source.y])
            path = self.find_interception_path(source, nearest_threat,distable)
            if path and len(path) > 1:
                if best is None or score>best[0]:
                    best=score,path
                
        if best:
            path=best[1]
            return (path[0], path[1])
        return None

    def find_interception_path(self, source: Point, threat: Point,distable):
        """寻找拦截路径"""
        # 简单实现：尝试找到从源点到威胁点的路径
        # 实际应用中可以使用更复杂的路径规划算法
        
        directions = [(0, 1), (0, -1), (1, 0), (-1, 0)]
        best_path = None
        
        min_distance = distable[(source.x,source.y)]
        # 尝试四个方向
        for dx, dy in directions:
            nx, ny = source.x + dx, source.y + dy
            if 0 <= nx < len(self.game_map) and 0 <= ny < len(self.game_map[0]):
                # 检查是否可通行
                tile = self.game_map[nx][ny]
                if tile.tile_type in [TileType.Mountain, TileType.Swamp]:
                    continue
                if tile.tile_type==TileType.City and tile.color_index!=self.color:
                    continue   
                if (nx,ny) not in distable:
                    continue
                # 计算到威胁的距离
                distance = distable[(nx,ny)]
                if distance < min_distance:
                    min_distance = distance
                    best_path = [source, Point(nx, ny)]
        
        return best_path
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

        self.y_offset,self.x_offset=self.upd_map_state()
        if not self.king_position:
            self.find_king_position()
        self.enemy_visable=False
        vise=[]
        for i in range(map_width):
            for j in range(map_height):
                if self.game_map[i][j].color_index!=self.color and self.game_map[i][j].color_index:
                    self.enemy_visable=True
                    vise.append((i,j))
        self.distab=self.bfs(vise)
        self.check_king_threat()


    def upd_map_state(self):
        map_width = len(self.game_map)
        map_height = len(self.game_map[0])
        tiles=np.zeros((map_width,map_height))
        armies=np.zeros((map_width,map_height))
        owner=np.zeros((map_width,map_height))
        cities=[]
        generals=[]
        for i in range(map_width):
            for j in range(map_height):
                tiles[i][j]=self.game_map[i][j].tile_type
                armies[i][j]=self.game_map[i][j].army_size
                if not self.game_map[i][j].color_index:
                    owner[i][j]=-1
                else:
                    owner[i][j]=self.game_map[i][j].color_index
                if self.game_map[i][j].tile_type==TileType.King:
                    generals.append((i,j))
                if self.game_map[i][j].tile_type==TileType.City:
                    cities.append((i,j))
        tiles, y_padding, x_padding = pad(tiles,TileType.Mountain)
        owner, y_padding, x_padding = pad(owner,-1)
        armies, y_padding, x_padding = pad(armies,0)
        y_offset = y_padding[0]
        x_offset = x_padding[0]
        map_state=self.game_state
        map_state[:,:,7] = (np.logical_or(tiles == TileType.Fog, tiles == TileType.Obstacle)).astype('float32')
        visible_tiles = map_state[:, :, 7] != 1
        map_state[:, :, 8] = np.logical_or(map_state[:, :, 8] == 1, map_state[:, :, 7] != 1).astype('float32')
        undiscovered_tiles = map_state[:, :, 8] != 1
        map_state[:,:,9] += 1
        map_state[visible_tiles, 9] = 0
        map_state[visible_tiles, 0] = owner[visible_tiles] == self.color
        map_state[visible_tiles, 1] = owner[visible_tiles] < 0 # Neutral
        map_state[undiscovered_tiles, 1] = 1 # Assume that all undiscovered tiles are neutral until discovered
        map_state[visible_tiles, 2] = owner[visible_tiles] !=self.color
        for y, x in cities:
            map_state[y+y_offset, x+x_offset, 5] = 1
        for y, x in generals:
            map_state[y+y_offset, x+x_offset, 6] = 1
        map_state[:, :, 3] = np.logical_or(tiles == TileType.Plain, owner >= 0) # Set empty tiles
        map_state[:,:,3] = np.logical_or(map_state[:,:,3], tiles == TileType.Fog)
        map_state[:, :, 4] = np.logical_or(tiles == TileType.Mountain, tiles == TileType.Obstacle)# Set mountains
        city_tiles = map_state[:, :, 5] == 1
        map_state[city_tiles, 4] = 0 # Ensure that cities in fog don't get marked as mountains
        map_state[city_tiles, 3] = 0 # Ensure that cities that are owned aren't marked as empty
        map_state[visible_tiles, 10] = armies[visible_tiles]
        self.game_state=map_state.astype('float32')
        #print(y_padding,x_padding)
        return y_padding,x_padding

    def evaluate_move(self, source: Point, direction: Tuple[int, int]):
        nx, ny = source.x + direction[0], source.y + direction[1]
        # 边界检查
        if nx < 0 or ny < 0 or nx >= len(self.game_map) or ny >= len(self.game_map[0]):
            return -1,-1
        
        target_tile = self.game_map[nx][ny]
        source_tile = self.game_map[source.x][source.y]
        move_army = source_tile.army_size - 1  # 可移动兵力
        # 2. 目标为山地
        if target_tile.tile_type == TileType.Mountain:  # 山地
            return -1,-1
        sc_mul=1
        # 4. 目标为敌方单位
        if target_tile.color_index != self.color:
            if target_tile.tile_type == TileType.King and move_army >= target_tile.army_size + 2:  # 敌方首都
                return 1000,1
            if target_tile.tile_type == TileType.City and move_army<=target_tile.army_size:
                return -1,-1
            if target_tile.tile_type == TileType.City and move_army>target_tile.army_size:
                sc_mul=2
                if target_tile.color_index is None:
                    sc_mul=1/math.exp(move_army/target_tile.army_size)
                    #sc_mul=target_tile.army_size/(target_tile.army_size+move_army)
                    if sc_mul<1/6:
                        sc_mul=-1
            if target_tile.tile_type != TileType.City:
                sc_mul=1.5
        if target_tile.color_index == self.color and target_tile.tile_type==TileType.King and move_army<=target_tile.army_size*2:
            return -1,-1
        dr=dtp2dir(direction)
        
        score=self.move_dir[dr][source.x][source.y]
        if self.enemy_visable and (nx,ny) in self.distab and self.turns_count>100:
            sd=self.distab[(source.x,source.y)]
            td=self.distab[(nx,ny)]
            if td<sd:
                sc_mul*=2
        return score,sc_mul
    def evaluate_move1(self, source: Point, direction: Tuple[int, int]) -> float:
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
        
        if target_tile.color_index == self.color and target_tile.tile_type==TileType.King and (move_army>=100 or (self.turns_count>=200 and random.random()<1/2)):
            return 0

        # 首都保护：前期减少移动首都兵力
        if source_tile.tile_type == TileType.King:
            score *= 0.2 if self.turns_count < 25 else 0.8
        ma_scalar=0.12
        if self.enemy_visable and (nx,ny) in self.distab and self.turns_count>100 and move_army>self.turns_count:
            sd=self.distab[(source.x,source.y)]
            td=self.distab[(nx,ny)]
            if td<sd:
                ma_scalar=0.2
        score += move_army * ma_scalar
        return score

    def handle_move(self):
        #[player.color, player.team, data.army, data.land]
        if not self.game_map or not self.init_game_info or not self.color:
            return
        map_width = len(self.game_map)
        map_height = len(self.game_map[0])
        y_offset=self.y_offset
        x_offset=self.x_offset
        self_land=0
        self_army=0
        for dat in self.leader_board_data:
            if dat[0]==self.color:
                self_land=dat[3]
                self_army=dat[2]
        max_army=0
        for i in range(len(self.game_map)):
            for j in range(len(self.game_map[0])):
                tile = self.game_map[i][j]
                if tile.color_index == self.color and tile.army_size > 1 and tile.tile_type!=TileType.King:
                    max_army=max(max_army,tile.army_size)
        if self.defense_mode and max_army<=100:
            defense_move = self.find_defense_move()
            if defense_move:
                source, target = defense_move
                print(f"Defense move: ({source.x},{source.y}) -> ({target.x},{target.y})")
                return ({"x": source.x, "y": source.y},{"x": target.x, "y": target.y},False)
        #if self.enemy_visable:
        move_half=False
        if self.collect_time>=60:
            self.game_mode=1
        if self.collect_time==0 and max_army>=self_army/6 and max_army>=100:
            self.game_mode=1
        if not (max_army>=self_army/6 and max_army>=100):
            self.game_mode=0
        if self.game_mode==1:
            # 收集所有可移动格子（兵力>1）
            lands = np.zeros((map_width,map_height))
            for i in range(len(self.game_map)):
                for j in range(len(self.game_map[0])):
                    tile = self.game_map[i][j]
                    if tile.color_index == self.color and tile.army_size >= 10:
                        lands[i][j]=1
            self.tile_pos,self.move_dir=self.model.infer(self.game_state)
            self.tile_pos=self.tile_pos[0,y_offset[0]:zero2none(-y_offset[1]),x_offset[0]:zero2none(-x_offset[1])]
            self.move_dir=self.move_dir[0,:,y_offset[0]:zero2none(-y_offset[1]),x_offset[0]:zero2none(-x_offset[1])]
            print(self.tile_pos.shape,lands.shape,self.move_dir.shape)
            self.tile_pos=np.multiply(self.tile_pos,lands)
            tidx=np.argmax(self.tile_pos)
            r,c=np.unravel_index(tidx,self.tile_pos.shape)

            self.rep_pen*=0.95
            # 评估所有可能移动
            directions = [(0, 1), (0, -1), (1, 0), (-1, 0)]
            source=Point(int(r),int(c))
            if source==self.king_position:
                move_half=True
            md=[]
            sc=[]
            for direction in directions:
                score,sc_mul = self.evaluate_move(source, direction)
                if score < 0:  # 跳过无效移动
                    continue
                #sc_mul*=self.evaluate_move1(source, direction)
                md.append(direction)
                sc.append(max(score-self.rep_pen[source.x][source.y][dtp2dir(direction)],0)*sc_mul)
            
            # 选择最佳移动
            if not md:
                return
            
            #direction = random.choices(md,sc)[0]
            direction = md[np.argmax(sc)]
            target_point = {"x": source.x + direction[0], "y": source.y + direction[1]}
            self.rep_pen[source.x][source.y][dtp2dir(direction)]+=0.3
        else:
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
                    score = self.evaluate_move1(source, direction)
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
            if source==self.king_position and self.turns_count>=200:
                move_half=True
            target_point = {"x": source.x + direction[0], "y": source.y + direction[1]}
        
        if self.game_map[target_point["x"]][target_point["y"]].color_index and self.game_map[target_point["x"]][target_point["y"]].color_index!=self.color:
            self.collect_time=0
        else:
            self.collect_time+=1

        return({"x": source.x, "y": source.y},target_point,move_half)