"""测试 server.py 的 WebSocket 接口"""
import asyncio, websockets, json

async def test():
    async with websockets.connect('ws://localhost:8000/ws') as ws:
        for i in range(3):
            msg = await asyncio.wait_for(ws.recv(), timeout=5)
            d = json.loads(msg)
            print(f'WS #{i+1}: frame={d.get("frame")} state={d.get("state")} '
                  f'risk={d.get("risk_score")} alert={d.get("alert_level")}')

asyncio.run(test())
print('WebSocket OK')
