#!/bin/bash
echo "启动生产服务器..."
gunicorn -w 4 -b 0.0.0.0:1214 --timeout 120 server:app