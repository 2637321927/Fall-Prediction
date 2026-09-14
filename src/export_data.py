"""导出 fall-01 数据 + 骨架动画到 artifacts/ 文件夹"""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.animation as anim
from pathlib import Path

# 项目根目录（本脚本位于 <项目>/src/）
BASE = Path(__file__).resolve().parent.parent

# 加载数据
k = np.load(BASE / 'data' / 'keypoints' / 'fall-01_keypoints.npy')
l = np.load(BASE / 'data' / 'labels' / 'fall-01_labels.npy')
l = l[:len(k)]

# 骨架连接 + 关节名
edges = [(0,1),(0,2),(1,3),(2,4),(5,6),(5,7),(7,9),(6,8),(8,10),
         (5,11),(6,12),(11,12),(11,13),(13,15),(12,14),(14,16)]
joint_names = ['鼻子','左眼','右眼','左耳','右耳','左肩','右肩',
               '左肘','右肘','左腕','右腕','左髋','右髋','左膝','右膝','左踝','右踝']
label_names = {0:'正常', 1:'失衡', 2:'倒地'}

out = BASE / 'artifacts'
out.mkdir(parents=True, exist_ok=True)

# === 1. 导出文本 ===
with open(out / 'fall-01_data.txt', 'w', encoding='utf-8') as f:
    f.write(f'fall-01 关键点数据\n')
    f.write(f'总帧数: {len(k)}, 关节数: 17, 通道: (x, y, conf)\n')
    f.write(f'标签分布: 正常={(l==0).sum()}, 失衡={(l==1).sum()}, 倒地={(l==2).sum()}\n')
    f.write('=' * 60 + '\n\n')
    for fi in range(len(k)):
        f.write(f'--- 帧 {fi}  标签={l[fi]} ({label_names[l[fi]]}) ---\n')
        f.write(f'{"关节":<6} {"x":>8} {"y":>8} {"conf":>8}\n')
        for j in range(17):
            x, y, c = k[fi, j]
            f.write(f'{joint_names[j]:<6} {x:8.4f} {y:8.4f} {c:8.4f}\n')
        f.write('\n')
print(f'文本: {out / "fall-01_data.txt"}')

# === 2. 骨架动画 ===
fig, ax = plt.subplots(figsize=(6, 8))
ax.set_xlim(-0.6, 0.6)
ax.set_ylim(-0.7, 0.7)
ax.invert_yaxis()
ax.set_aspect('equal')
ax.set_xlabel('x (normalized)')
ax.set_ylabel('y (normalized)')
ax.set_title('URFD fall-01 Skeleton Animation')

scat = ax.scatter([], [], s=80, c='blue', zorder=3)
line_objs = [ax.plot([], [], 'gray', lw=2)[0] for _ in edges]
t_text = ax.text(0.02, 0.98, '', transform=ax.transAxes, fontsize=12, va='top')
l_text = ax.text(0.02, 0.93, '', transform=ax.transAxes, fontsize=14, va='top', fontweight='bold')

colors_map = {0: 'green', 1: 'orange', 2: 'red'}

def init():
    scat.set_offsets(np.empty((0, 2)))
    for ln in line_objs:
        ln.set_data([], [])
    return [scat] + line_objs + [t_text, l_text]

def update(frame):
    pts = k[frame, :, :2]
    cf = k[frame, :, 2]
    mask = cf > 0.3
    scat.set_offsets(pts)
    scat.set_facecolor(['blue' if m else 'lightgray' for m in mask])
    for i, (a, b) in enumerate(edges):
        if cf[a] > 0.3 and cf[b] > 0.3:
            line_objs[i].set_data([pts[a, 0], pts[b, 0]], [pts[a, 1], pts[b, 1]])
        else:
            line_objs[i].set_data([], [])
    t_text.set_text(f'Frame: {frame}/{len(k)-1}  ({frame/30:.1f}s)')
    lbl = label_names[l[frame]]
    l_text.set_text(lbl)
    l_text.set_color(colors_map[l[frame]])
    return [scat] + line_objs + [t_text, l_text]

print('生成动画 (160帧)...')
ani = anim.FuncAnimation(fig, update, frames=len(k), init_func=init, blit=True, interval=33)
ani.save(str(out / 'fall-01_animation.gif'), writer='pillow', fps=10, dpi=100)
plt.close()
print(f'视频: {out / "fall-01_animation.gif"}')
print('完成!')
