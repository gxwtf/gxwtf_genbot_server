#!/usr/bin/env python3
"""
机器人平台服务器
"""

import os
import importlib
from multiprocessing import Process
from flask import Flask, request, jsonify
import time
from bot import gen_api
from dotenv import load_dotenv

load_dotenv()

app = Flask(__name__)

# 从环境变量获取服务器URL
SERVER_URL = os.getenv('SERVER_URL', 'https://api.generals.gxwtf.cn')
# 从环境变量获取端口号，默认为1214
PORT = int(os.getenv('PORT', 1214))

# 可用的机器人种类列表
AVAILABLE_BOT_TYPES = ['AIgbot', 'Gbot', 'Kongbot']

room_bots = {}  # 格式: {room_id: {bot_type: {bot_number: bot_info}}}
proc_list = []  # 存储进程对象

# 添加跨域请求支持
@app.after_request
def after_request(response):
    """添加跨域请求头"""
    response.headers.add('Access-Control-Allow-Origin', '*')
    response.headers.add('Access-Control-Allow-Headers', 'Content-Type,Authorization')
    response.headers.add('Access-Control-Allow-Methods', 'GET,PUT,POST,DELETE,OPTIONS')
    return response

# 处理OPTIONS请求（预检请求）
@app.route('/', methods=['OPTIONS'])
@app.route('/<path:path>', methods=['OPTIONS'])
def options_handler(path=None):
    """处理跨域预检请求"""
    response = jsonify({'success': True})
    response.headers.add('Access-Control-Allow-Origin', '*')
    response.headers.add('Access-Control-Allow-Headers', 'Content-Type,Authorization')
    response.headers.add('Access-Control-Allow-Methods', 'GET,PUT,POST,DELETE,OPTIONS')
    return response

def get_next_bot_number(room_id, bot_type):
    """获取下一个机器人编号"""
    if room_id not in room_bots:
        return 1
    if bot_type not in room_bots[room_id]:
        return 1
    # 找到当前最大的编号
    existing_numbers = list(room_bots[room_id][bot_type].keys())
    if not existing_numbers:
        return 1
    return max(existing_numbers) + 1

def add_bot_process(room_id, bot_type):
    """
    添加并启动一个机器人进程
    """
    print('获取到请求')
    try:
        # 动态导入机器人模块
        bot_module = importlib.import_module(f"bot.{bot_type}")
        
        # 获取下一个编号
        bot_number = get_next_bot_number(room_id, bot_type)
        username = f"{bot_type}_{bot_number}"
        
        # 创建机器人实例
        gbot = bot_module.GBot(room_id, username)
        
        # 创建并启动进程（移除延迟参数）
        p = Process(target=gen_api.run_bot, args=(gbot, SERVER_URL, 0))
        p.start()
        
        # 存储进程信息（不存储Process对象本身，只存储基本信息）
        bot_info = {
            'room_id': room_id,
            'bot_type': bot_type,
            'username': username,
            'bot_number': bot_number,
            'process_id': p.pid,
            'start_time': time.time(),
            'process_alive': True  # 手动跟踪进程状态
        }
        
        # 更新数据结构
        if room_id not in room_bots:
            room_bots[room_id] = {}
        if bot_type not in room_bots[room_id]:
            room_bots[room_id][bot_type] = {}
        
        room_bots[room_id][bot_type][bot_number] = bot_info
        proc_list.append(p)
        
        return True, f"成功添加机器人 {username} 到房间 {room_id}"
        
    except Exception as e:
        return False, f"添加机器人失败: {e}"

def is_process_alive(pid):
    """检查进程是否存活"""
    try:
        os.kill(pid, 0)
        return True
    except (OSError, ProcessLookupError):
        return False

def remove_last_bot_by_type(room_id, bot_type):
    """
    移除指定房间和类型的最后一个机器人
    """
    if room_id not in room_bots:
        return False, "房间不存在"
    
    if bot_type not in room_bots[room_id]:
        return False, f"房间 {room_id} 中没有 {bot_type} 类型的机器人"
    
    bot_type_dict = room_bots[room_id][bot_type]
    if not bot_type_dict:
        return False, f"房间 {room_id} 中没有 {bot_type} 类型的机器人"
    
    # 找到最大的编号（最后加入的）
    last_bot_number = max(bot_type_dict.keys())
    bot_info = bot_type_dict[last_bot_number]
    
    # 终止进程
    if is_process_alive(bot_info['process_id']):
        try:
            os.kill(bot_info['process_id'], 9)  # 发送SIGKILL信号
        except (OSError, ProcessLookupError):
            pass  # 进程可能已经结束
    
    # 清理记录
    del bot_type_dict[last_bot_number]
    
    # 如果该类型没有机器人了，清理类型字典
    if not bot_type_dict:
        del room_bots[room_id][bot_type]
    
    # 如果房间没有机器人了，清理房间字典
    if not room_bots[room_id]:
        del room_bots[room_id]
    
    return True, f"已移除房间 {room_id} 中 {bot_type} 类型的最后一个机器人 ({bot_info['username']})"

def remove_all_bots_in_room(room_id):
    """
    移除指定房间的所有机器人
    """
    if room_id not in room_bots:
        return False, f"房间 {room_id} 不存在"
    
    room_data = room_bots[room_id]
    removed_count = 0
    
    # 收集所有需要终止的进程
    processes_to_kill = []
    
    for bot_type, bots_dict in room_data.items():
        for bot_number, bot_info in bots_dict.items():
            # 终止进程
            if is_process_alive(bot_info['process_id']):
                try:
                    os.kill(bot_info['process_id'], 9)  # 发送SIGKILL信号
                    processes_to_kill.append(bot_info['process_id'])
                except (OSError, ProcessLookupError):
                    pass  # 进程可能已经结束
            removed_count += 1
    
    # 清理数据结构
    del room_bots[room_id]
    
    # 从proc_list中移除对应的进程对象
    global proc_list
    proc_list = [p for p in proc_list if p.pid not in processes_to_kill]
    
    return True, f"已移除房间 {room_id} 中的所有机器人，共 {removed_count} 个"

@app.route('/add/', methods=['GET', 'OPTIONS'])
def add_bot():
    """
    添加机器人接口
    GET参数: roomId, type
    """
    if request.method == 'OPTIONS':
        return jsonify({'success': True})
    
    room_id = request.args.get('roomId')
    bot_type = request.args.get('type', 'AIgbot')
    
    if not room_id:
        return jsonify({'success': False, 'message': '缺少roomId参数'}), 400
    
    success, message = add_bot_process(room_id, bot_type)
    
    if success:
        return jsonify({'success': True, 'message': message})
    else:
        return jsonify({'success': False, 'message': message}), 500

@app.route('/status/', methods=['GET', 'OPTIONS'])
def get_global_status():
    """
    获取全局机器人状态
    """
    if request.method == 'OPTIONS':
        return jsonify({'success': True})
    
    status_info = []
    total_bots = 0
    
    for room_id, room_data in room_bots.items():
        room_info = {
            'room_id': room_id,
            'bot_types': {}
        }
        
        for bot_type, bots_dict in room_data.items():
            type_info = []
            for bot_number, bot_info in bots_dict.items():
                alive = is_process_alive(bot_info['process_id'])
                type_info.append({
                    'username': bot_info['username'],
                    'bot_number': bot_info['bot_number'],
                    'alive': alive,
                    'process_id': bot_info['process_id'],
                    'uptime': int(time.time() - bot_info['start_time'])
                })
                total_bots += 1
            
            room_info['bot_types'][bot_type] = type_info
        
        status_info.append(room_info)
    
    return jsonify({
        'success': True,
        'total_bots': total_bots,
        'total_rooms': len(status_info),
        'rooms': status_info
    })

@app.route('/status/<room_id>/', methods=['GET', 'OPTIONS'])
def get_room_status(room_id):
    """
    获取指定房间的机器人状态
    """
    if request.method == 'OPTIONS':
        return jsonify({'success': True})
    
    if room_id not in room_bots:
        return jsonify({'success': False, 'message': f'房间 {room_id} 不存在'}), 404
    
    room_info = {
        'room_id': room_id,
        'bot_types': {}
    }
    
    for bot_type, bots_dict in room_bots[room_id].items():
        type_info = []
        for bot_number, bot_info in bots_dict.items():
            alive = is_process_alive(bot_info['process_id'])
            type_info.append({
                'username': bot_info['username'],
                'bot_number': bot_info['bot_number'],
                'alive': alive,
                'process_id': bot_info['process_id'],
                'uptime': int(time.time() - bot_info['start_time'])
            })
        
        room_info['bot_types'][bot_type] = type_info
    
    return jsonify({
        'success': True,
        'room': room_info
    })

@app.route('/remove/', methods=['GET', 'OPTIONS'])
def remove_bot():
    """
    移除机器人接口
    GET参数: roomId, type（可选）
    - 如果同时提供roomId和type：移除指定房间和类型的最后一个机器人
    - 如果只提供roomId：移除指定房间的所有机器人
    """
    if request.method == 'OPTIONS':
        return jsonify({'success': True})
    
    room_id = request.args.get('roomId')
    bot_type = request.args.get('type')
    
    if not room_id:
        return jsonify({'success': False, 'message': '缺少roomId参数'}), 400
    
    # 如果没有提供type参数，移除整个房间的所有机器人
    if not bot_type:
        success, message = remove_all_bots_in_room(room_id)
        if success:
            return jsonify({'success': True, 'message': message})
        else:
            return jsonify({'success': False, 'message': message}), 404
    
    # 如果提供了type参数，按原逻辑移除指定类型的最后一个机器人
    success, message = remove_last_bot_by_type(room_id, bot_type)
    
    if success:
        return jsonify({'success': True, 'message': message})
    else:
        return jsonify({'success': False, 'message': message}), 404

@app.route('/type/', methods=['GET', 'OPTIONS'])
def get_bot_types():
    """
    获取所有可用的机器人种类
    """
    if request.method == 'OPTIONS':
        return jsonify({'success': True})
    
    return jsonify({
        'success': True,
        'bot_types': AVAILABLE_BOT_TYPES,
        'count': len(AVAILABLE_BOT_TYPES)
    })

@app.route('/shutdown/', methods=['GET', 'OPTIONS'])
def shutdown():
    """
    关闭服务器并结束所有机器人
    """
    if request.method == 'OPTIONS':
        return jsonify({'success': True})
    
    end_all()
    return jsonify({'success': True, 'message': '服务器正在关闭...'})

def end_all():
    """
    结束所有机器人进程 - 立即终止所有进程
    """
    print("开始立即关闭所有机器人...")
    
    # 直接终止所有进程，不再等待退出房间
    print("立即终止所有机器人进程...")
    for p in proc_list:
        if p.is_alive():
            try:
                p.terminate()  # 发送terminate信号
            except Exception as e:
                print(f"终止进程时出错: {e}")
    
    # 等待很短时间让进程终止
    print("等待进程终止...")
    for p in proc_list:
        try:
            p.join(timeout=1)  # 只等待1秒
        except Exception as e:
            print(f"等待进程终止时出错: {e}")
    
    # 如果还有存活的进程，强制杀死
    for p in proc_list:
        if p.is_alive():
            try:
                print(f"强制杀死进程 {p.pid}")
                os.kill(p.pid, 9)  # 发送SIGKILL信号
            except Exception as e:
                print(f"强制杀死进程时出错: {e}")
    
    # 清理数据结构
    room_bots.clear()
    proc_list.clear()
    print("所有机器人已立即关闭")

if __name__ == '__main__':
    print("启动机器人平台服务器...")
    print(f"服务器URL: {SERVER_URL}")
    print(f"端口号: {PORT}")
    print("可用接口:")
    print("  GET /type/                        - 获取所有可用的机器人种类")
    print("  GET /add/?roomId=3&type=AIgbot    - 添加机器人到指定房间")
    print("  GET /status/                      - 查看全局机器人状态")
    print("  GET /status/<room_id>/            - 查看指定房间机器人状态")
    print("  GET /remove/?roomId=3             - 移除指定房间所有的机器人")
    print("  GET /remove/?roomId=3&type=AIgbot - 移除指定房间类型的最后一个机器人")
    print("  GET /shutdown/                    - 关闭服务器")
    
    try:
        app.run(host='localhost', port=PORT, debug=False)
    except KeyboardInterrupt:
        print("\n收到中断信号，正在关闭服务器...")
    finally:
        end_all()
        print("服务器已关闭")