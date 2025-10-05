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

1. `GET /add/?roomId=2&type=aigbot`  添加机器人到指定房间
2. `GET /status/` 查看全局机器人状态
3. `GET /status/<room_id>/` 查看指定房间机器人状态
4. `GET /remove/?roomId=2&type=aigbot` 移除指定房间类型的最后一个机器人
5. `GET /shutdown/` 关闭服务器
