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


---