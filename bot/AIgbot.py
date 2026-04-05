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
    def __hash__(self):
        return hash((self.x, self.y))
    def __str__(self):
        return f"Point({self.x}, {self.y})"
ORIGINAL_MAP_WIDTH=23
MAP_CHANNELS=11
def pad(state, fill_value = 0, map_width = ORIGINAL_MAP_WIDTH):
    if map_width<state.shape[1] or map_width<state.shape[0]:
        return state,(0,0),(0,0)
    x_diff = float(map_width - state.shape[1]) / 2
    x_padding = (math.ceil(x_diff), math.floor(x_diff))
    y_diff = float(map_width - state.shape[0]) / 2
    y_padding = (math.ceil(y_diff), math.floor(y_diff))
    #print("hhhh --", y_padding, x_padding, state.shape)
    return np.pad(state, pad_width=(y_padding, x_padding), 
                mode='constant', constant_values=fill_value), y_padding, x_padding

def unpad(x, pad_width):
    slices = []
    for c in pad_width:
        e = None if c[1] == 0 else -c[1]
        slices.append(slice(c[0], e))
    return x[tuple(slices)]

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

class GeneralPrediction:
    """敌人将军位置预测系统"""
    def __init__(self):
        self.fog_scores = {}  # 迷雾格子得分
        self.enemy_appearances = []  # 敌人单位出现的位置历史
        self.predicted_general = None  # 预测的将军位置
        self.confidence = 0  # 预测置信度
        
    def score_decay(self,game_map,color):
        for pt in self.fog_scores:
            if game_map[pt.x][pt.y].tile_type == TileType.Fog or game_map[pt.x][pt.y].tile_type == TileType.Obstacle:
                continue
            if game_map[pt.x][pt.y].tile_type!=TileType.King and (game_map[pt.x][pt.y].tile_type!=TileType.City or game_map[pt.x][pt.y].color_index is None or game_map[pt.x][pt.y].color_index==color):
                self.fog_scores[pt]*=0.5
    def update_prediction(self, enemy_positions, game_map, map_width, map_height, game_state, x_offset, y_offset,color):
        """更新将军位置预测"""
        # 记录敌人单位出现的位置
        for pos in enemy_positions:
            if pos[0] not in self.enemy_appearances:
                self.enemy_appearances.append(pos[0])
                #print(f"Enemy appeared at: {pos}")

        gs=unpad(game_state,(y_offset, x_offset))
        #gs=game_state[y_offset:y_offset+map_width,x_offset:x_offset+map_height]
        for i in range(map_width):
            for j in range(map_height):
                if self.fog_scores and gs[i][j][6]==1 and game_map[i][j].color_index!=color:
                    if Point(i,j) not in self.fog_scores:
                        self.fog_scores[Point(i,j)]=0
                    self.fog_scores[Point(i,j)]=max(1000,self.fog_scores[Point(i,j)])

        # 对每个敌人出现的位置，使用BFS对周围的迷雾格子进行评分
        for enemy_pos in enemy_positions:
            self._bfs_score_fog(enemy_pos, game_map, map_width, map_height,gs,max_distance=8)
            
        self.score_decay(game_map,color)
        #print(self.fog_scores)
        # 找到得分最高的迷雾格子作为预测的将军位置
        if self.fog_scores:
            self.predicted_general = max(self.fog_scores.keys(), key=lambda p: self.fog_scores[p])
            self.confidence = self.fog_scores[self.predicted_general]
            print(f"Predicted general at: {self.predicted_general} with confidence: {self.confidence}")
        else:
            self.predicted_general = None
            self.confidence = 0
            
    def _bfs_score_fog(self, start: Tuple[Point,int], game_map, map_width, map_height, game_state, max_distance):
        """从敌人出现位置BFS遍历迷雾并评分"""
        distances = {}
        visited = set()
        queue = deque()
        
        distances[start[0]] = 0
        visited.add(start[0])
        queue.append(start[0])
        val=min(2+start[1]**0.8,60)

        directions = [(0, 1), (0, -1), (1, 0), (-1, 0)]
        
        while queue:
            current = queue.popleft()
            current_dist = distances[current]
            #print(current,current=start[0])
            if game_state[current.x][current.y][8] != 1:
                # 分数基于距离：距离越近，分数越高
                if current not in self.fog_scores:
                    self.fog_scores[current] = 0
                self.fog_scores[current] += 3*val//(current_dist+1)

            # 如果达到最大距离，停止搜索
            if current_dist >= max_distance or (game_map[current.x][current.y].tile_type!=TileType.Fog and game_map[current.x][current.y].tile_type!=TileType.Obstacle and current!=start[0]):
                continue
                
            for d in directions:
                nx, ny = current.x + d[0], current.y + d[1]
                
                # 检查边界
                if nx < 0 or nx >= map_width or ny < 0 or ny >= map_height:
                    continue
                    
                nb = Point(nx, ny)
                # 跳过已访问的节点
                if nb in visited:
                    continue
                # 获取邻居瓦片
                neighbor_tile = game_map[nx][ny]
                # 跳过不可通行的地形
                if neighbor_tile.tile_type in [TileType.Mountain, TileType.Obstacle]:
                    continue
                # 添加到已访问集合
                visited.add(nb)
                distances[nb] = current_dist + 1
                queue.append(nb)
       
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
        self.general_predictor = GeneralPrediction()
        self.previous_game_map = None
        self.previous_leaderboard = None
        self.dis2pg=None
        self.disable_ai=False
        self.cities_cnt=[]
        self.should_or_not=False
        self.centre_dis=None

    def init_map(self, map_width: int, map_height: int):
        self.game_map = [
            [TileProp(TileType.Fog, None, None) for _ in range(map_height)]
            for _ in range(map_width)
        ]
        self.previous_game_map = [
            [TileProp(TileType.Fog, None, None) for _ in range(map_height)]
            for _ in range(map_width)
        ]
        self.game_state=np.zeros((ORIGINAL_MAP_WIDTH, ORIGINAL_MAP_WIDTH, MAP_CHANNELS)).astype('float32')
        self.rep_pen=np.zeros((map_width,map_height,4))
        self.king_position=None
        self.defense_mode = False
        self.collect_time=0
        self.game_mode=0
        self.dis2pg=None
        self.general_predictor = GeneralPrediction()
        if map_height>23 or map_height>23:
            self.disable_ai=True
            self.game_state=np.zeros((map_width,map_height, MAP_CHANNELS)).astype('float32')
        self.previous_leaderboard = None
        self.cities_cnt=[[1]*16,[0]*16]
        self.should_or_not=False
        
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
        for dx in range(-10, 10):
            for dy in range(-10, 10):
                    
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
        for dx in range(-10, 10):
            for dy in range(-10, 10):
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
    
    def detect_enemy_appearances(self):
        """检测敌人单位出现的位置（新增检测逻辑）"""
        enemy_positions = []
        if not self.previous_game_map:
            return enemy_positions
            
        map_width = len(self.game_map)
        map_height = len(self.game_map[0])
        
        for i in range(map_width):
            for j in range(map_height):
                prev_tile = self.previous_game_map[i][j]
                curr_tile = self.game_map[i][j]
                
                # 情况1：从迷雾中出现敌人
                if ((prev_tile.tile_type==TileType.Fog or prev_tile.tile_type==TileType.Obstacle) and 
                    curr_tile.color_index != self.color and 
                    curr_tile.color_index is not None and curr_tile.army_size is not None):
                    enemy_positions.append((Point(i, j),curr_tile.army_size))
                    print(f"Enemy emergence detected at {i},{j}: {curr_tile.army_size}")
                    continue
                
                # 情况2：敌人兵力突然大幅增加
                if (prev_tile.color_index != self.color and prev_tile.color_index is not None and prev_tile.color_index >= 0 and
                    curr_tile.color_index != self.color and curr_tile.color_index is not None and curr_tile.color_index >= 0):
                    if prev_tile.army_size is None or curr_tile.army_size is None:
                        continue
                    # 计算上一回合自身及相邻格子的总兵力
                    #print(prev_tile.color_index,curr_tile.color_index)
                    total_prev = prev_tile.army_size
                    for dx, dy in [(0,1),(0,-1),(1,0),(-1,0)]:
                        ni, nj = i + dx, j + dy
                        if 0 <= ni < map_width and 0 <= nj < map_height:
                            neighbor = self.previous_game_map[ni][nj]
                            if neighbor.color_index != self.color and neighbor.color_index is not None and neighbor.army_size is not None:
                                total_prev += neighbor.army_size
                    if curr_tile.army_size > total_prev+10:
                        enemy_positions.append((Point(i, j),curr_tile.army_size - total_prev))
                        print(f"Enemy reinforcement detected at {i},{j}: {curr_tile.army_size} > {total_prev}")
        
        return enemy_positions
    
    def patch_map(self, map_diff: List[Union[int, TilePropTuple]]):
        if not self.game_map:
            return
        # 保存当前地图到上一回合
        if self.game_map:
            map_width = len(self.game_map)
            map_height = len(self.game_map[0])
            self.previous_game_map = [
                [TileProp(self.game_map[i][j].tile_type, self.game_map[i][j].color_index, self.game_map[i][j].army_size) 
                 for j in range(map_height)] 
                for i in range(map_width)
            ]
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
        last_turn_vis=self.enemy_visable
        self.enemy_visable=False
        vise=[]
        for i in range(map_width):
            for j in range(map_height):
                if self.game_map[i][j].color_index!=self.color and self.game_map[i][j].color_index:
                    self.enemy_visable=True
                    vise.append((i,j))
        if self.enemy_visable:
            # 检测敌人出现
            enemy_appearances = self.detect_enemy_appearances()
            # 更新将军预测
            if enemy_appearances:
                self.general_predictor.update_prediction(enemy_appearances, self.game_map, map_width, map_height,self.game_state,self.x_offset,self.y_offset,self.color)
            if not last_turn_vis:
                #self.send_message(f"看到你了{self.game_map[vise[0][0]][vise[0][1]].color_index}")
                pass
        
        self.general_predictor.score_decay(self.game_map,self.color)
        self.distab=self.bfs(vise)
        self.check_king_threat()
        if self.turns_count%50!=0:
            for i in range(len(self.leader_board_data)):
                if self.leader_board_data[i][0]==self.color:
                    self.cities_cnt[0][self.color]=0
                    for j in range(map_width):
                        for k in range(map_height):
                            if self.game_map[j][k].color_index==self.color and (self.game_map[j][k].tile_type==TileType.King or self.game_map[j][k].tile_type==TileType.City):
                                self.cities_cnt[0][self.color]+=1
                    continue
                delta=self.leader_board_data[i][2]-self.previous_leaderboard[i][2]
                if delta>=self.cities_cnt[0][self.leader_board_data[i][0]] or self.cities_cnt[1][self.leader_board_data[i][0]]>=20 and delta<50:
                    self.cities_cnt[0][self.leader_board_data[i][0]]=delta
                    self.cities_cnt[1][self.leader_board_data[i][0]]=0
                self.cities_cnt[1][self.leader_board_data[i][0]]+=1
        self.previous_leaderboard=self.leader_board_data


        if self.turns_count==1:
            si=0
            sj=0
            for i in range(map_width):
                for j in range(map_height):
                    if self.game_map[i][j].tile_type==TileType.Fog and abs(2*i-map_width)+abs(2*j-map_height)<abs(2*si-map_width)+abs(2*sj-map_height):
                        si=i
                        sj=j
            self.centre_dis=self.bfs([(si,sj)])

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

    def roast(self,multi):
        p="们" if multi else ""
        rl=["局势对你方很不妙",f"再这样下去，你{p}就要失败了","点击输入文本","点左上角箭头可以投降","游戏教程在主页右上角","今天走路来的啊","池塘中有10朵莲，我只采1朵",f"你{p}和平原有个相同的特点","你知道喝茶要找什么吗","50的水瓶我45就买到了","?","如何集中注意力？让你做事更集中的7个小技巧","「回家睡觉去吧」的意思是指人们应该回家安心休息。随着现代生活节奏的加快，越来越多的人晚上都处于忙碌状态，导致失眠、缺乏睡眠等问题逐渐突出。而回家睡觉不仅可以让身体得到充分休息，还可以促进身心健康。因此，这句话不仅是一种劝告，更是传递出对健康生活的呼吁。","你可是我的快乐源泉啊！"]
        self.send_message(random.choice(rl))

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
        if self.general_predictor.predicted_general:
            current_dist = self.dis2pg.get((source.x,source.y),1000)
            new_dist = self.dis2pg.get((nx,ny),1000)
            if new_dist < current_dist:
                if self.general_predictor.confidence<1000:
                    sc_mul*=2
                else:
                    sc_mul*=2.5
        return score,sc_mul
    def evaluate_move1(self, source: Point, direction: Tuple[int, int],self_army:int) -> float:
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
        if not target_tile.color_index or target_tile.color_index <= 0:
            if target_tile.tile_type == TileType.City:  # 中立要塞
                if self.should_or_not:
                    if self.general_predictor.predicted_general:
                        g_dist = self.dis2pg.get((nx,ny),1000)
                        k_dist=abs(nx-self.king_position.x)+abs(ny-self.king_position.y)
                        if g_dist>=k_dist*1.5:
                            return 30+df/6 if move_army >= target_tile.army_size + 2 else 0

                    return 25+df/6 if move_army >= target_tile.army_size + 2 else 0
                
                if self.enemy_visable:
                    return 15+df/6 if move_army >= target_tile.army_size + 2 else 0
                else:
                    return 20+df/6 if move_army >= target_tile.army_size + 2 else 0
            if self.enemy_visable and (nx,ny) in self.distab and self.turns_count>100 and move_army>self.turns_count/8:
                sd=self.distab[(source.x,source.y)]
                td=self.distab[(nx,ny)]
                if td<sd:
                    return 15+move_army/3
            if not self.enemy_visable:
                return 15+move_army/3
            return 15
        
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
        if self.turns_count<=100 and not self.enemy_visable:
            if self.centre_dis.get((nx,ny),abs(2*nx-len(self.game_map))+abs(2*ny-len(self.game_map[0])))<self.centre_dis.get((source.x,source.y),abs(2*source.x-len(self.game_map))+abs(2*source.y-len(self.game_map[0]))):
                score+=move_army*0.5

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
        if self.general_predictor.predicted_general:
            current_dist = self.dis2pg.get((source.x,source.y),1000)
            new_dist = self.dis2pg.get((nx,ny),1000)
            if new_dist < current_dist:
                #if move_army>self_army/8 and move_army>=self.turns_count/4:
                if move_army>self_army/8:
                    score += move_army * 0.15
        return score

    def handle_move(self):
        #[player.color, player.team, data.army, data.land]
        if not self.game_map or not self.init_game_info or not self.color:
            return
        
        # calculate cland and carmy
        playercnt=len(self.leader_board_data)
        cland = []
        carmy = []
        for i in range(playercnt):
            clr=self.leader_board_data[i][0]
            if self.leader_board_data[i][2]==0:
                cland.append(0)
                carmy.append(0)
                continue
            cland.append(self.leader_board_data[i][3] + 25 * (self.cities_cnt[0][clr] - 1))
            carmy.append(self.leader_board_data[i][2] + 48 * (self.cities_cnt[0][clr] - 1))

        
        map_width = len(self.game_map)
        map_height = len(self.game_map[0])
        y_offset=self.y_offset
        x_offset=self.x_offset
        self_land=0
        self_army=0
        win_rate=0.5
        max_ecities=0
        self_cland=0
        maxe_cland=0
        for i,dat in enumerate(self.leader_board_data):
            if dat[0]==self.color:
                self_land=dat[3]
                self_army=dat[2]
                win_rate=carmy[i]/sum(carmy)
                self_cland=cland[i]
            else:
                max_ecities=max(max_ecities,self.cities_cnt[0][dat[0]])
                maxe_cland=max(maxe_cland,cland[i])
        # if self.turns_count%25==0:
        #     print("win rate:",win_rate)
        # for debug
        # if self.turns_count%50==0 and self.turns_count>0:
        #     self.send_message("win rate "+format(win_rate, '.2f'))
        
        if self.turns_count%25==0 and win_rate>=0.65:
            self.roast(len(self.leader_board_data)>2)

        max_army=0
        
        for i in range(len(self.game_map)):
            for j in range(len(self.game_map[0])):
                tile = self.game_map[i][j]
                if tile.color_index == self.color and tile.army_size > 1 and tile.tile_type!=TileType.King:
                    max_army=max(max_army,tile.army_size)
        if self.general_predictor.predicted_general:
            self.dis2pg=self.bfs([(self.general_predictor.predicted_general.x,self.general_predictor.predicted_general.y)])
        #check king
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
                score = self.evaluate_move1(source, direction,self_army)
                if score==1000:
                    target = Point(source.x + direction[0],source.y + direction[1])
                    return ({"x": source.x, "y": source.y},{"x": target.x, "y": target.y},False)
        
        if self.defense_mode:
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
        if self.disable_ai:
            self.game_mode=0

        if win_rate > 0.43 and self.cities_cnt[0][self.color] < max_ecities: 
            self.should_or_not=True
        elif win_rate > 0.47 and maxe_cland > self_cland + 10 and self_cland + (50 - self.turns_count % 50) < maxe_cland: 
            self.should_or_not=True
        elif win_rate > 0.47 and maxe_cland > self_cland + 20: 
            self.should_or_not=True
        else:
            self.should_or_not=False
        # for dat in self.leader_board_data:
        #     print("color",dat[0],':',self.cities_cnt[0][dat[0]])
        
        
        if self.turns_count<=20:
            return

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
            self.rep_pen*=0.95
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
                    score = self.evaluate_move1(source, direction,self_army)
                    if score < 0:  # 跳过无效移动
                        continue
                    moves.append((source, direction, score*(1-self.rep_pen[source.x][source.y][dtp2dir(direction)])))
            
            # 选择最佳移动
            if not moves:
                return
            
            best_moves = []
            max_score = max(moves, key=lambda x: x[2])[2]
            for move in moves:
                if move[2] == max_score:
                    best_moves.append(move)
            
            source, direction, _ = random.choice(best_moves)
            self.rep_pen[source.x][source.y][dtp2dir(direction)]+=0.3
            if source==self.king_position and self.turns_count>=200:
                move_half=True
            target_point = {"x": source.x + direction[0], "y": source.y + direction[1]}
        
        if self.game_map[target_point["x"]][target_point["y"]].color_index and self.game_map[target_point["x"]][target_point["y"]].color_index!=self.color:
            self.collect_time=0
        else:
            self.collect_time+=1

        return ({"x": source.x, "y": source.y},target_point,move_half)
