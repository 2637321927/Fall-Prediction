import csv, re, numpy as np, sys
from collections import defaultdict
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent   # 项目根目录
sys.path.insert(0, str(Path(__file__).resolve().parent))

csv_data = defaultdict(list)
with open(BASE / 'data' / 'raw' / 'prevfall' / 'keypoints_no confidence.csv') as f:
    r = csv.DictReader(f)
    for row in r:
        if row['id'].startswith('CGFM2'):
            csv_data[row['id']].append(row)

csv_fns = sorted([int(re.search(r'_(\d+)_', fid).group(1)) for fid in csv_data.keys()])
print(f'CSV: {len(csv_fns)} frames (range {csv_fns[0]}~{csv_fns[-1]})')

k = np.load(BASE / 'data' / 'variants' / 'prevfall_keypoints_CGFM2.npy')
print(f'Ours: {len(k)} frames')

# Compare middle frame
mid_fn = csv_fns[len(csv_fns)//2]
csv_fid = [fid for fid in csv_data.keys() if f'_{mid_fn}_' in fid][0]
csv_row = csv_data[csv_fid][0]

b25_cols = ['nose','leye','reye','lear','rear','lshoulder','rshoulder',
            'lelbow','relbow','lwrist','rwrist','lhip','rhip','lknee','rknee','lankle','rankle']
names = ['nose ','Leye ','Reye ','Lear ','Rear ','Lsho ','Rsho ',
         'Lelb ','Relb ','Lwri ','Rwri ','Lhip ','Rhip ','Lkne ','Rkne ','Lank ','Rank ']

hx = (float(csv_row['lhipx'])+float(csv_row['rhipx']))/2
hy = (float(csv_row['lhipy'])+float(csv_row['rhipy']))/2
head_y = float(csv_row['nosey'])
ankle_y = max(float(csv_row['lankley']), float(csv_row['rankley']))
h = ankle_y - head_y
print(f'Official height: {h:.0f}px')

print(f'{"Joint":6s} {"Off_x":>8s} {"Off_y":>8s} {"Our_x":>8s} {"Our_y":>8s} {"Diff_x":>7s} {"Diff_y":>7s} {"Conf":>5s}')

diffs_x, diffs_y = [], []
for j in range(17):
    col = b25_cols[j]
    ox, oy = float(csv_row[col+'x']), float(csv_row[col+'y'])
    mx, my, mc = k[mid_fn, j]
    onx = (ox-hx)/h if h>20 else 0
    ony = (oy-hy)/h if h>20 else 0

    if mc > 0.3:
        dx, dy = abs(mx-onx), abs(my-ony)
        diffs_x.append(dx); diffs_y.append(dy)
        print(f'{names[j]:6s} {onx:8.3f} {ony:8.3f} {mx:8.3f} {my:8.3f} {dx:7.3f} {dy:7.3f} {mc:5.2f}')
    else:
        print(f'{names[j]:6s} {onx:8.3f} {ony:8.3f} {mx:8.3f} {my:8.3f} {"-":>7s} {"-":>7s} {mc:5.2f}')

print(f'\nAvg diff: x={np.mean(diffs_x):.4f}  y={np.mean(diffs_y):.4f}')

# Check face points specifically
print('\n--- Face points detail ---')
for j in [0,1,2,3,4]:
    col = b25_cols[j]
    ox, oy = float(csv_row[col+'x']), float(csv_row[col+'y'])
    mx, my, mc = k[mid_fn, j]
    onx = (ox-hx)/h; ony = (oy-hy)/h
    print(f'{names[j]} off=({onx:.3f},{ony:.3f}) ours=({mx:.3f},{my:.3f}) conf={mc:.2f}')
