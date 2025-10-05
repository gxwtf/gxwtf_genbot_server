import socketio
from typing import List, Tuple, Union
import time


class TileType:
    King = 0 # base
    City = 1 # spawner
    Fog = 2 #it's color unit = null
    Obstacle = 3 #Either City or Mountain, which is unknown, it's color unit = null
    Plain = 4 #blank , plain, Neutral, 有数值时，即是army
    Mountain = 5
    Swamp = 6

TilePropTuple = Tuple[int, int, int]

class TileProp:
    def __init__(
        self, tile_type: int, color_index: Union[int, None], army_size: Union[int, None]
    ):
        self.tile_type = tile_type
        self.color_index = color_index  # owner_id
        self.army_size = army_size


class GBotBase:
    def __init__(self, room_id: str, username: str):
        self.room_id = room_id
        self.username = username
        self.my_player_id = None
        self.color = None
        self.init_game_info = None
        self.game_map = None
        self.turns_count = 0 
        self.leader_board_data = None

    def init_map(self, map_width: int, map_height: int):
        self.game_map = [
            [TileProp(TileType.Fog, None, None) for _ in range(map_height)]
            for _ in range(map_width)
        ]
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
    def handle_move(self):
        # should return ({"x": source.x, "y": source.y},{"x": target.x, "y": target.y},move_half)
        pass
    
#sio=None

def init_bot(gbot,server_url):
    sio = socketio.Client()

    @sio.event
    def connect():
        print(f"socket client connect to server: {sio.sid}")

    @sio.event
    def update_room(room: dict):
        print("update_room")
        #print(room)
        botplayer=list(filter(lambda p:p['id']==gbot.my_player_id,room["players"]))[0]
        if botplayer['isRoomHost']:
            humanplayer=list(filter(lambda p:"bot" not in p['username'].lower(),room["players"]))
            if humanplayer:
                sio.emit("change_host",humanplayer[0]["id"])
        gbot.color = next(
            (p["color"] for p in room["players"] if p["id"] == gbot.my_player_id), None
        )

    @sio.event
    def set_player_id(player_id: str):
        print(f"set_player_id: {player_id}")
        gbot.my_player_id = player_id

    @sio.event
    def error(title: str, message: str):
        print("GET ERROR FROM SERVER:\n", title, message)

    @sio.event
    def room_message(player: dict, message: str):
        print(f"room_message: {player['username']} {message}")

    @sio.event
    def game_started(init_game_info: dict):
        print("Game started:", init_game_info)
        gbot.init_game_info = init_game_info
        gbot.init_map(init_game_info["mapWidth"], init_game_info["mapHeight"])

    @sio.event
    def attack_failure(from_p, to, message: str):
        print(f"attack_failure: {from_p} {to} {message}")

    @sio.event
    def game_update(
        map_diff: List[Union[int, TilePropTuple]],
        turns_count: int,
        leader_board_data: dict,
    ):
        print(f"game_update: {turns_count}")
        gbot.turns_count = turns_count  # 更新回合数
        gbot.leader_board_data=leader_board_data
        gbot.patch_map(map_diff)
        
        if sio.connected:
            res = gbot.handle_move()
            sio.emit("attack", res)
        else:
            print("连接已断开，跳过攻击操作")
        #leader_board_data is a list of [player.color, player.team, data.army, data.land]
        #print(gbot.color,leader_board_data)

    @sio.event
    def game_over(captured_by: dict):
        print(f"game_over: {captured_by['username']}")
        # 不再自动断开连接，等待下一局游戏
        print("游戏结束，等待下一局游戏...")
        
    @sio.event
    def game_ended(winner: dict, replay_link: str):
        print(f"game_ended: {winner[0]['username']} {replay_link}")
        # 不再自动断开连接，等待下一局游戏
        print("游戏结束，等待下一局游戏...")
        sio.disconnect()

    sio.connect(
        server_url + f"?username={gbot.username}&roomId={gbot.room_id}"
    )
    #sio.wait()
    return sio

def start_bot(sio):
    sio.emit("force_start")
    sio.wait()

def run_bot(gbot, server_url, wait_time=0):
    """运行机器人并保持连接以支持多局游戏"""
    while True:  # 添加循环以支持多局游戏
        try:
            sio = init_bot(gbot, server_url)
            time.sleep(wait_time)
            start_bot(sio)
        except Exception as e:
            print(f"机器人运行出错: {e}")
            time.sleep(wait_time)

def leave_room(gbot, server_url):
    """退出房间"""
    try:
        sio = socketio.Client()
        
        @sio.event
        def connect():
            print(f"连接服务器准备退出房间: {sio.sid}")
            # 发送退出房间的请求
            sio.emit("leave_room")
            time.sleep(1)  # 等待服务器处理
            sio.disconnect()
        
        @sio.event
        def disconnect():
            print(f"已断开连接，退出房间完成")
        
        sio.connect(server_url + f"?username={gbot.username}&roomId={gbot.room_id}")
        sio.wait()
        
    except Exception as e:
        print(f"退出房间时出错: {e}")