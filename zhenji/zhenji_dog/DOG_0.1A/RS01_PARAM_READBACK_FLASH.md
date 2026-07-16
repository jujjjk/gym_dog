# RS01 参数读回固件刷写

本固件在原有96字节电机快照和批量CAN基础上增加：

- SPI操作码 `0x13`：请求读取一个RS01参数。
- SPI响应魔数 `0x5B`：返回type-17参数响应。
- capability bit1：参数读回可用；更新后 `board_capabilities=3`。

两块STM32使用同一份 `Core/Src/main.c`，但板角色必须分别编译。

## Keil构建

打开 `MDK-ARM/DOG_0.1.uvprojx`。

### A板

1. Options for Target → C/C++ → Define 加入 `BOARD_IS_A=1`。
2. Output名称设置为 `DOG_A`，勾选Create HEX File。
3. Rebuild all target files。
4. 将生成的A板固件刷入连接SPI0、管理电机 `0x11–0x23` 的STM32。

### B板

1. 将Define改为 `BOARD_IS_A=0`，不能保留冲突的 `BOARD_IS_A=1`。
2. Output名称设置为 `DOG_B`。
3. Rebuild all target files，不能复用A板目标文件缓存。
4. 将生成的B板固件刷入连接SPI1、管理电机 `0x31–0x43` 的STM32。

刷写完成后将两块STM32和全部电机断电10秒再重新上电。

## Jetson验证

启动新版 `text/app.py` 后：

```bash
curl -s http://127.0.0.1:8000/api/state | \
  python3 -c 'import json,sys; d=json.load(sys.stdin); print(d["0x11"]["board_capabilities"], d["0x31"]["board_capabilities"])'
```

必须输出：

```text
3 3
```

参数读回单项检查（十进制索引：`0x700B=28683`、`0x7018=28696`）：

```bash
curl -s 'http://127.0.0.1:8000/api/rs04/read_param_f32?motor_id=17&index=28683'
curl -s 'http://127.0.0.1:8000/api/rs04/read_param_f32?motor_id=17&index=28696'
```

未刷任一块板时，5730策略会拒绝启动并由服务发送全电机STOP。
