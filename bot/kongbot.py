import random
import logging
from .gen_api import *
from typing import List, Tuple, Union, Dict, Any
from collections import deque
import multiprocessing as mp
from .kongbot_backend import start_game
import time
import sys
import os

def conv_tt(x,y):
    if x is not None:
        return x
    if y==TileType.Fog:
        return -3
    if y==TileType.Obstacle:
        return -4
    if y==TileType.Plain or y==TileType.City:
        return -1
    if y==TileType.Mountain:
        return -2

class Point:
    def __init__(self, x: int, y: int):
        self.x = x
        self.y = y

# 简化的机器人工作函数
# 简化的机器人工作函数
def bot_worker(conn, start_data, color):
    """
    在单独进程中运行的机器人逻辑
    """
    # 创建一个简化的Generals类，只包含机器人需要的功能
    class BotGenerals:
        def __init__(self, conn, start_data, color):
            self.conn = conn
            self._start_data = start_data
            self.color = color
            self.game_map = []
            self._seen_update = False
            
            # 初始化地图
            if start_data:
                self.init_map(start_data['mapWidth'], start_data['mapHeight'])
        
        def init_map(self, map_width, map_height):
            self.game_map = [TileProp(TileType.Fog, None, None) for _ in range(map_width * map_height)]
            
        def move(self, x1, y1, x2, y2, move_half=False):
            if not self._seen_update:
                raise ValueError("Cannot move before first map seen")
            # 通过连接发送移动指令
            try:
                self.conn.send(('move', ({"x": x1, "y": y1}, {"x": x2, "y": y2}, move_half)))
            except:
                pass
            
        def get_updates(self):
            """生成器函数，返回游戏更新"""
            while True:
                try:
                    # 从连接接收更新
                    if self.conn.poll(1.0):  # 1秒超时
                        msg = self.conn.recv()
                        if msg == 'STOP':  # 终止信号
                            break
                        if msg[0] == 'update':
                            self._seen_update = True  # 设置已收到更新
                            yield msg[1]
                except:
                    break
                    
        def _make_update(self, map_diff, turns_count, leader_board_data):
            """处理游戏更新数据"""
            # 更新地图状态
            i = j = 0
            for diff in map_diff:
                if isinstance(diff, int):
                    j += diff
                else:
                    if j < len(self.game_map):
                        self.game_map[j] = TileProp(*diff)
                    j += 1

            rows, cols = self._start_data['mapWidth'], self._start_data['mapHeight']
            self._seen_update = True
            _cities = []
            genpos = [-1] * 16
            lands = [0] * 16
            armies = [0] * 16
            
            for i in range(rows * cols):
                if i < len(self.game_map) and self.game_map[i].tile_type == TileType.City:
                    _cities.append(i)
                if i < len(self.game_map) and self.game_map[i].tile_type == TileType.King:
                    genpos[self.game_map[i].color_index] = i
                    
            for s in leader_board_data:
                if s[0] < len(lands):
                    lands[s[0]] = s[3]
                if s[0] < len(armies):
                    armies[s[0]] = s[2]
                
            return {
                'complete': False,
                'rows': rows,
                'cols': cols,
                'player_index': self.color,
                'turn': turns_count,
                'army_grid': [[self.game_map[y*cols + x].army_size if y*cols + x < len(self.game_map) and self.game_map[y*cols + x].army_size is not None else 0
                              for x in range(cols)]
                              for y in range(rows)],
                'tile_grid': [[conv_tt(self.game_map[y*cols + x].color_index, self.game_map[y*cols + x].tile_type) 
                              if y*cols + x < len(self.game_map) else -3
                              for x in range(cols)]
                              for y in range(rows)],
                'lands': lands,
                'armies': armies,
                'alives': [s[2] > 0 for s in leader_board_data] + 16 * [False],
                'generals': [(-1, -1) if g == -1 else (g // cols, g % cols)
                             for g in genpos],
                'cities': [(c // cols, c % cols) for c in _cities],
            }
    
    # 创建机器人实例并运行
    client = BotGenerals(conn, start_data, color)
    try:
        start_game(client)
    except Exception as e:
        logging.error(f"Bot process error: {e}")
    finally:
        try:
            conn.close()
        except:
            pass

class Generals(object):
    def __init__(self,ctx=None):
        logging.debug("Creating connection")
        
        self._start_data = {}
        logging.debug("Joining game")

        self.room=None
        self.color=None
        self.my_player_id=None
        self.game_map = []
        self._seen_update = False
        self._stars = []
        
        # 使用多进程上下文
        if ctx is None:
            ctx = mp.get_context('spawn')
        self.ctx = ctx
        
        # 使用管道而不是队列
        self.parent_conn, self.child_conn = ctx.Pipe()
        self._bot_process = None
        
    def init_map(self, map_width: int, map_height: int):
        self.game_map = [TileProp(TileType.Fog, None, None) for _ in range(map_height) for _ in range(map_width)]

    def move(self, x1, y1, x2, y2, move_half=False):
        if not self._seen_update:
            raise ValueError("Cannot move before first map seen")
        # 移动指令现在通过process_update发送

    def start_bot(self):
        """启动机器人进程"""
        if self._bot_process is None or not self._bot_process.is_alive():
            # 准备初始数据
            start_data = {
                'mapWidth': self._start_data['mapWidth'],
                'mapHeight': self._start_data['mapHeight']
            }
            
            # 启动机器人进程
            self._bot_process = self.ctx.Process(
                target=bot_worker,
                args=(self.child_conn, start_data, self.color)
            )
            self._bot_process.daemon = True
            self._bot_process.start()

    def stop_bot(self):
        """停止机器人进程"""
        if self._bot_process and self._bot_process.is_alive():
            # 发送终止信号
            try:
                self.parent_conn.send('STOP')
            except:
                pass
            self._bot_process.join(timeout=1.0)
            if self._bot_process.is_alive():
                self._bot_process.terminate()
            try:
                self.parent_conn.close()
            except:
                pass

    def process_update(self, map_diff, turns_count, leader_board_data):
        """处理游戏更新，并向机器人进程发送更新数据"""
        # 更新地图状态
        i = j = 0
        for diff in map_diff:
            if isinstance(diff, int):
                j += diff
            else:
                if j < len(self.game_map):
                    self.game_map[j] = TileProp(*diff)
                j += 1

        rows, cols = self._start_data['mapWidth'], self._start_data['mapHeight']
        self._seen_update = True
        _cities = []
        genpos = [-1] * 16
        lands = [0] * 16
        armies = [0] * 16
        
        for i in range(rows * cols):
            if i < len(self.game_map) and self.game_map[i].tile_type == TileType.City:
                _cities.append(i)
            if i < len(self.game_map) and self.game_map[i].tile_type == TileType.King:
                genpos[self.game_map[i].color_index] = i
                
        for s in leader_board_data:
            if s[0] < len(lands):
                lands[s[0]] = s[3]
            if s[0] < len(armies):
                armies[s[0]] = s[2]
            
        update_data = {
            'complete': False,
            'rows': rows,
            'cols': cols,
            'player_index': self.color,
            'turn': turns_count,
            'army_grid': [[self.game_map[y*cols + x].army_size if y*cols + x < len(self.game_map) and self.game_map[y*cols + x].army_size is not None else 0
                          for x in range(cols)]
                          for y in range(rows)],
            'tile_grid': [[conv_tt(self.game_map[y*cols + x].color_index, self.game_map[y*cols + x].tile_type) 
                          if y*cols + x < len(self.game_map) else -3
                          for x in range(cols)]
                          for y in range(rows)],
            'lands': lands,
            'armies': armies,
            'alives': [s[2] > 0 for s in leader_board_data] + 16 * [False],
            'generals': [(-1, -1) if g == -1 else (g // cols, g % cols)
                         for g in genpos],
            'cities': [(c // cols, c % cols) for c in _cities],
        }
        
        # 将更新数据通过管道发送给机器人进程
        try:
            if self._bot_process and self._bot_process.is_alive():
                self.parent_conn.send(('update', update_data))
        except Exception as e:
            logging.error(f"Failed to send update to bot process: {e}")
        
        return update_data

    def handle_move(self):
        """处理移动请求"""
        try:
            # 检查是否有移动指令
            if self.parent_conn.poll(0.1):  # 0.1秒超时
                msg = self.parent_conn.recv()
                if msg[0] == 'move':
                    return msg[1]
        except:
            pass
        return None

    def close(self):
        self.stop_bot()

class GBot(GBotBase):
    def __init__(self, room_id: str, username: str = "GenniaBot"):
        # 使用 get_context 来创建多进程上下文
        try:
            ctx = mp.get_context('fork')
        except:
            ctx = mp.get_context()
            
        super().__init__(room_id, username)
        self.color = None
        self.init_game_info = None
        self.game_map = None
        self.turns_count = 0
        
        # 使用上下文创建 Generals 实例
        self.genobj = Generals(ctx=ctx)
        # 设置必要的初始数据
        self.genobj.color = self.color
    def init_map(self, map_width: int, map_height: int):
        self.game_map = [
            [TileProp(TileType.Fog, None, None) for _ in range(map_height)]
            for _ in range(map_width)
        ]
        self.on_game_start(self.init_game_info)

    def on_game_start(self, data: Dict[str, Any]):
        """处理游戏开始事件"""
        #super().on_game_start(data)
        self.genobj._start_data=data
        # self.genobj._start_data = {
        #     'mapWidth': data['map_width'],
        #     'mapHeight': data['map_height']
        # }
        self.genobj.init_map(data['mapWidth'], data['mapHeight'])
        self.genobj.color = self.color
        # 启动机器人进程
        self.genobj.start_bot()

    def patch_map(self, map_diff: List[Union[int, TilePropTuple]]):
        """处理地图更新"""
        if not self.game_map:
            return
            
        # 处理更新
        self.genobj.process_update(map_diff, self.turns_count, self.leader_board_data)
        
        # 更新本地游戏地图状态
        map_width = len(self.game_map)
        map_height = len(self.game_map[0])
        flattened = [tile for row in self.game_map for tile in row]
        new_state = [[None for _ in range(map_height)] for _ in range(map_width)]
        i = j = 0
        for diff in map_diff:
            if isinstance(diff, int):
                j += diff
            else:
                if j < len(flattened):
                    flattened[j] = TileProp(*diff)
                j += 1
        for i in range(map_width):
            for j in range(map_height):
                idx = i * map_height + j
                if idx < len(flattened):
                    new_state[i][j] = flattened[idx]
        self.game_map = new_state

    def check_king_threat(self):
        """检查国王威胁级别"""
        king_position=Point(self.init_game_info['king']["x"],self.init_game_info['king']["y"])
        map_height=len(self.game_map)
        king_army=self.game_map[king_position.x][king_position.y].army_size        
        
        # 检查国王周围5格内是否有敌方单位
        for dx in range(-8, 9):
            for dy in range(-8, 9):
                    
                nx, ny = king_position.x + dx, king_position.y + dy
                if 0 <= nx < len(self.game_map) and 0 <= ny < len(self.game_map[0]):
                    tile = self.game_map[nx][ny]
                    if tile.color_index != self.color and tile.color_index is not None and tile.army_size is not None:
                        # 计算威胁级别：敌方兵力 + 距离权重
                        distance = abs(dx) + abs(dy)
                        if tile.army_size-2*distance>=king_army:
                            return True
        return False

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
                return 15+df/6 if move_army >= target_tile.army_size + 2 else 0
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
        
        score += move_army * ma_scalar
        return score
    def handle_move(self):
        """处理移动请求"""
        kres=self.genobj.handle_move()
        if self.check_king_threat():
            return kres
        cities_cnt=0
        self_army=0
        for dat in self.leader_board_data:
            if dat[0]==self.color:
                self_army=dat[2]
        max_army=0
        for i in range(len(self.game_map)):
            for j in range(len(self.game_map[0])):
                tile = self.game_map[i][j]
                if tile.color_index == self.color and tile.tile_type==TileType.City:
                    cities_cnt+=1
                if tile.color_index == self.color and tile.army_size > 1 and tile.tile_type!=TileType.King:
                    max_army=max(max_army,tile.army_size)
        if cities_cnt<(self_army-400)/100 and self_army<=self.turns_count*3 and max_army<=self_army/6:
            print("taking cities")
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
        return kres

    def close(self):
        """清理资源"""
        self.genobj.close()
        super().close()
