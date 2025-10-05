# gxwtf_genbot_server

配置环境变量

```bash
cp .env.example .env
```

端口号默认为 1214

## 开发环境

```bash
./run_dev.sh
```

## 生产环境

```bash
./run_prod.sh
```

## 接口

1. `GET /type/` 获取所有可用的机器人种类
2. `GET /add/?roomId=2&type=aigbot`  添加机器人到指定房间
3. `GET /status/` 查看全局机器人状态
4. `GET /status/<room_id>/` 查看指定房间机器人状态
5. `GET /remove/?roomId=2&type=aigbot` 移除指定房间类型的最后一个机器人
6. `GET /shutdown/` 关闭服务器
