# Jetson 电机通信服务

从 `jetson@172.19.54.119:/home/jetson/text` 同步，2026-09-29。
包含 FastAPI HTTP 服务、SPI 驱动、Unix socket RT 协议/服务及当前网页。
源码逻辑按 Jetson 保存（规范化换行、清理 app.py 末尾空白）；SHA256 见 ../mydog_ros2_ws/JETSON_SYNC_20260929.json。

现有设备运行位置是 `/home/jetson/text`。Python 依赖包括 fastapi、uvicorn、pydantic、spidev。
`systemd/` 保存当前用户服务和两个覆盖配置：状态后台刷新100Hz，CPU亲和性1/2/3、Nice=-5。
B23500 控制/发送仍为50Hz，单独的绑核参数由 ../mydog_ros2_ws/start_capture.sh 提供。
服务文件仍引用现有设备的目录；新机器安装时按实际路径调整 WorkingDirectory。

本次同步未安装/重启服务、未执行使能或步行。用户服务文件需由部署者安装至
`~/.config/systemd/user/`，覆盖配置放入同名 `.service.d/` 目录后才能生效。
