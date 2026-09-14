# 跌倒检测 API 文档

实时跌倒检测服务的 HTTP/WebSocket 接口说明。

## 快速开始

```bash
# 启动服务（视频文件模式）
python server.py --checkpoint ../checkpoints/best_model.pt --video ../video/FDFSM5.mp4

# 启动服务（摄像头模式）
python server.py --checkpoint ../checkpoints/best_model.pt

# 自定义端口/监听所有网卡（供前端访问）
python server.py --checkpoint ../checkpoints/best_model.pt --host 0.0.0.0 --port 8000
```

启动后服务地址：`http://<host>:<port>`（默认 `0.0.0.0:8000`）

---

## 接口总览

| 方法 | 路径 | 说明 |
|------|------|------|
| GET | `/api/status` | 当前检测状态快照 |
| GET | `/api/history?limit=N` | 历史检测记录 |
| GET | `/api/config` | 获取阈值配置 |
| POST | `/api/config` | 修改阈值配置 |
| WS | `/ws` | WebSocket 实时推送 |
| GET | `/video` | MJPEG 实时视频流 |

---

## 1. GET /api/status

当前最新检测结果快照。

**响应示例：**

```json
{
  "online": true,
  "fps": 12.5,
  "thresholds": {
    "low": 0.60,
    "high": 0.75,
    "immediate": 0.85
  },
  "frame": 394,
  "state": "UNSTABLE",
  "risk_score": 0.786,
  "alert_level": "HIGH_RISK",
  "probabilities": {
    "normal": 0.21,
    "unstable": 0.74,
    "falling": 0.05
  },
  "timestamp": 1724400000.123
}
```

**字段说明：**

| 字段 | 类型 | 说明 |
|------|------|------|
| `online` | bool | 推理是否在运行（视频播完=false） |
| `fps` | float | 实际推理帧率 |
| `thresholds` | object | 当前预警阈值 |
| `frame` | int | 当前帧号 |
| `state` | string | 模型判断状态：`NORMAL`/`UNSTABLE`/`FALLING` |
| `risk_score` | float | 风险分数 0.0~1.0（=P不稳+P摔倒） |
| `alert_level` | string | 预警等级：`SAFE`/`LOW_RISK`/`HIGH_RISK`/`IMMEDIATE` |
| `probabilities` | object | 三分类概率（和为1） |
| `timestamp` | float | Unix 时间戳（秒） |

---

## 2. GET /api/history

查询历史检测记录。

**参数：**

| 参数 | 类型 | 默认 | 说明 |
|------|------|------|------|
| `limit` | int | 100 | 返回最近 N 条记录（最大10000） |

**响应示例：**

```json
{
  "total": 579,
  "records": [
    {
      "frame": 0,
      "state": "NORMAL",
      "risk_score": 0.002,
      "alert_level": "SAFE",
      "probabilities": {"normal": 0.998, "unstable": 0.002, "falling": 0.000},
      "timestamp": 1724400000.000
    },
    {
      "frame": 394,
      "state": "UNSTABLE",
      "risk_score": 0.786,
      "alert_level": "HIGH_RISK",
      "probabilities": {"normal": 0.21, "unstable": 0.74, "falling": 0.05},
      "timestamp": 1724400000.123
    }
  ]
}
```

**字段说明：**

| 字段 | 说明 |
|------|------|
| `total` | 缓冲区内的总记录数 |
| `records` | 最近 limit 条记录，每条结构与 status 一致 |

---

## 3. GET /api/config

获取当前预警阈值配置。

**响应示例：**

```json
{
  "low": 0.60,
  "high": 0.75,
  "immediate": 0.85
}
```

**阈值含义（连续帧规则）：**

| 阈值 | 规则 |
|------|------|
| `low` | risk_score 连续3帧 > low → LOW_RISK |
| `high` | risk_score 连续2帧 > high → HIGH_RISK |
| `immediate` | risk_score 单帧 > immediate → IMMEDIATE |

---

## 4. POST /api/config

动态修改预警阈值（实时生效，无需重启）。

**请求体**（可只传要修改的项）：

```json
{
  "low": 0.50,
  "high": 0.70,
  "immediate": 0.80
}
```

**响应：**

```json
{
  "ok": true,
  "thresholds": {
    "low": 0.50,
    "high": 0.70,
    "immediate": 0.80
  }
}
```

---

## 5. WebSocket /ws

实时推送检测结果，前端订阅后被动接收。

**连接：** `ws://<host>:<port>/ws`

**推送频率：** 每 0.5 秒一条

**消息格式（JSON）：**

```json
{
  "online": true,
  "fps": 12.5,
  "frame": 396,
  "state": "FALLING",
  "risk_score": 0.868,
  "alert_level": "IMMEDIATE",
  "probabilities": {
    "normal": 0.02,
    "unstable": 0.11,
    "falling": 0.87
  },
  "timestamp": 1724400000.456
}
```

---

## 6. GET /video

MJPEG 实时视频流，画面叠加了骨架 + 风险条 + 状态文字。

**注意：** 无人访问时服务端不消耗编码资源（按需编码）。

**前端示例（HTML）：**

```html
<img src="http://localhost:8000/video" width="640" alt="实时检测画面">
```

---

## 完整调用示例

### cURL

```bash
# 当前状态
curl http://localhost:8000/api/status

# 历史（最近50条）
curl "http://localhost:8000/api/history?limit=50"

# 获取配置
curl http://localhost:8000/api/config

# 修改配置（调低预警阈值，更敏感）
curl -X POST http://localhost:8000/api/config \
  -H "Content-Type: application/json" \
  -d '{"low": 0.5, "high": 0.7, "immediate": 0.8}'

# 视频流（浏览器直接看）
curl http://localhost:8000/video
```

### Python (requests)

```python
import requests

status = requests.get("http://localhost:8000/api/status").json()
print(status["state"], status["risk_score"], status["alert_level"])

requests.post("http://localhost:8000/api/config", json={"immediate": 0.8})
```

### Python (websocket)

```python
import asyncio, json
import websockets

async def listen():
    async with websockets.connect("ws://localhost:8000/ws") as ws:
        while True:
            data = json.loads(await ws.recv())
            print(data["frame"], data["state"], data["risk_score"])

asyncio.run(listen())
```

---

## 状态机规则速查

```
risk_score > immediate(0.85) 单帧      → IMMEDIATE  (立即报警)
risk_score > high(0.75) 连续2帧        → HIGH_RISK  (高风险)
risk_score > low(0.60) 连续3帧         → LOW_RISK   (低风险)
否则                                  → SAFE       (安全)
```

`state`（NORMAL/UNSTABLE/FALLING）是模型 argmax 判断的具体状态，`alert_level` 是由风险分数和阈值推导的预警等级，两者信息互补。
