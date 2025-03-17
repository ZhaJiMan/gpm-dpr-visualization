from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np
from frykit.calc import region_mask
from matplotlib.patches import Rectangle

# 目录路径
dirpath_data = Path("../data")
dirpath_fig = Path("../fig")
if not dirpath_fig.exists():
    dirpath_fig.mkdir(parents=True)

# 设置数据范围和横截面起止点
filepath = next(dirpath_data.glob("*DPR.V8*"))
extents = [136.5, 137, 26.5, 28]
p0 = (135.5, 27.2)
p1 = (136.7, 27.2)

# 读取数据
with h5py.File(str(filepath), "r") as f:
    lon = f["NS/Longitude"][:]
    lat = f["NS/Latitude"][:]

# 截取数据
nscan, nray = lon.shape
mid = nray // 2
mask = region_mask(lon[:, mid], lat[:, mid], extents, apply_and=True)
index = np.s_[mask, :20]
lon = lon[index]
lat = lat[index]

# 构造表示横截面的直线
npt = 10
box = 0.1
line = np.column_stack((np.linspace(p0[0], p1[0], npt), np.linspace(p0[1], p1[1], npt)))

# 绘制数据点、直线上的采样点、和做平均的小方格
fig, ax = plt.subplots(figsize=(8, 8))
ax.set_aspect("equal")
ax.xaxis.set_visible(False)
ax.yaxis.set_visible(False)

ax.plot(lon.flat, lat.flat, ".", ms=5, c="k", label="data point")
ax.plot(line[:, 0], line[:, 1], lw=2, c="r")
ax.plot(line[:, 0], line[:, 1], ".", ms=10, c="r", label="cross section")

half = box / 2
for xc, yc in line:
    x0 = xc - half
    y0 = yc - half
    patch = Rectangle((x0, y0), box, box, fc="none", ec="r", lw=1)
    ax.add_patch(patch)
ax.legend(fontsize="x-large", markerscale=2)
fig.savefig(str(dirpath_fig / "diagram.png"), dpi=300, bbox_inches="tight")
plt.close(fig)
