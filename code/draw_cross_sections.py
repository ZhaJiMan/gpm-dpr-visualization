from pathlib import Path

import h5py
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm
from matplotlib.ticker import MultipleLocator, AutoMinorLocator
import cartopy.crs as ccrs
from cartopy.feature import LAND, OCEAN
import frykit.plot as fplt
from frykit.calc import region_ind
import cmaps

class CrossSection:
    '''
    对GPM二级产品求横截面的类.

    横截面直线上每个采样点对应一个小方框, 落入框内的数据点的平均值作为该采样点的值.
    将地图视为笛卡尔坐标系. 无法处理跨越180经度线的情况.
    '''
    def __init__(self, lon, lat, start, end, npt=100, box=0.2):
        '''
        Parameters
        ----------
        lon : (nscan, nray) ndarray
            二维经度数组.

        lat : (nscan, nray) ndarray
            二维纬度数组.

        start : 2 tuple of float
            横截面的起点.

        end : 2 tuple of float
            横截面的终点.

        npt : int
            横截面直线的采样点数(包含端点). 默认为100.

        box : float
            采样点对应的方框大小. 数值越大计算出的横截面越平滑.
            默认为0.2度(DPR的水平分辨率5km约为0.05度).

        Attributes
        ----------
        line : (npt, 2) ndarray
            横截面直线上每个采样点的经纬度.
        '''
        self.npt = npt
        self.box = box
        self.start = start
        self.end = end
        self.line = np.column_stack((
            np.linspace(start[0], end[0], npt),
            np.linspace(start[1], end[1], npt)
        ))

        # 收集落入每个box的数据点的mask.
        half = box / 2
        self.masks = []
        for xc, yc in self.line:
            x0 = xc - half
            x1 = xc + half
            y0 = yc - half
            y1 = yc + half
            mask = region_ind(lon, lat, [x0, x1, y0, y1])
            self.masks.append(mask)

    def __call__(self, data):
        '''
        计算横截面.

        Parameters
        ----------
        data : (nscan, nray) or (nscan, nray, nbin) ndarray
            二维或三维的数组.

        Returns
        -------
        cross : (npt,) or (nbin, npt) ndarray
            横截面. 缺测值用NaN填充.
        '''
        if data.ndim == 2:
            cross = np.full(self.npt, np.nan)
        elif data.ndim == 3:
            cross = np.full((self.npt, data.shape[2]), np.nan)
        else:
            raise ValueError('data是二维或三维数组')

        # 计算每个box内的平均值.
        for i, mask in enumerate(self.masks):
            subset = data[mask]
            if not np.isnan(subset).all():
                cross[i] = np.nanmean(subset, axis=0)

        # 二维横截面做转置方便画图.
        if cross.ndim == 2:
            cross = cross.T

        return cross

    def get_xticks(self, ntick=6, lon_formatter=None, lat_formatter=None):
        '''返回截面图所需的横坐标, 刻度位置和刻度标签.'''
        return fplt.get_cross_section_xticks(
            self.line[:, 0], self.line[:, 1], ntick,
            lon_formatter, lat_formatter
        )

def get_label_pos(start, end, pad=0.5):
    '''根据两点连线计算起点旁标签的位置.'''
    x0, y0 = start
    x1, y1 = end
    d = np.hypot(x1 - x0, y1 - y0)
    r = pad / d
    x = (1 + r) * x0 - r * x1
    y = (1 + r) * y0 - r * y1

    return x, y

def get_date_str(filepath):
    '''获取日期字符串.'''
    date_str = filepath.name.split('.')[4][:8]
    date_str = '-'.join([date_str[:4], date_str[4:6], date_str[6:]])

    return date_str

def read_GMI_data(filepath):
    '''读取GMI数据, 并用经纬度方框截取.'''
    with h5py.File(str(filepath), 'r') as f:
        Longitude = f['S1/Longitude'][:]
        Latitude = f['S1/Latitude'][:]
        surfacePrecipitation = f['S1/surfacePrecipitation'][:]

    # 截取数据.
    npixel = Longitude.shape[1]
    mid = npixel // 2
    mask = region_ind(Longitude[:, mid], Latitude[:, mid], extents)
    Longitude = Longitude[mask, :]
    Latitude = Latitude[mask, :]
    surfacePrecipitation = surfacePrecipitation[mask, :]
    surfacePrecipitation[surfacePrecipitation <= -9999] = np.nan

    return {
        'Longitude': Longitude,
        'Latitude': Latitude,
        'surfacePrecipitation': surfacePrecipitation
    }

def read_DPR_data(filepath):
    '''读取DPR数据, 并用经纬度方框截取.'''
    with h5py.File(str(filepath), 'r') as f:
        Longitude = f['NS/Longitude'][:]
        Latitude = f['NS/Latitude'][:]
        elevation = f['NS/PRE/elevation'][:]
        heightZeroDeg = f['NS/VER/heightZeroDeg'][:]
        precipRateNearSurface = f['NS/SLV/precipRateNearSurface'][:]
        precipRate = f['NS/SLV/precipRate'][:]

    # 截取数据.
    nray = Longitude.shape[1]
    mid = nray // 2
    mask = region_ind(Longitude[:, mid], Latitude[:, mid], extents)
    Longitude = Longitude[mask, :]
    Latitude = Latitude[mask, :]
    elevation = elevation[mask, :]
    heightZeroDeg = heightZeroDeg[mask, :]
    precipRateNearSurface = precipRateNearSurface[mask, :]
    precipRate = precipRate[mask, :, ::-1]  # 倒转高度维.

    # 设置缺测.
    for data in [elevation, heightZeroDeg, precipRateNearSurface, precipRate]:
        data[data <= -9999.9] = np.nan
    # 高度单位改为km.
    elevation /= 1000
    heightZeroDeg /= 1000

    # 设置DPR的高度.
    nbin = 176
    dh = 0.125
    height = (np.arange(nbin) + 0.5) * dh

    return {
        'Longitude': Longitude,
        'Latitude': Latitude,
        'elevation': elevation,
        'heightZeroDeg': heightZeroDeg,
        'precipRateNearSurface': precipRateNearSurface,
        'precipRate': precipRate,
        'height': height
    }

def read_LH_data(filepath):
    '''读取CSH或SLH数据, 并用经纬度方框截取.'''
    with h5py.File(str(filepath), 'r') as f:
        Longitude = f['Swath/Longitude'][:]
        Latitude = f['Swath/Latitude'][:]
        latentHeating = f['Swath/latentHeating'][:]

    # 截取数据.
    nray = Longitude.shape[1]
    mid = nray // 2
    mask = region_ind(Longitude[:, mid], Latitude[:, mid], extents)
    Longitude = Longitude[mask, :]
    Latitude = Latitude[mask, :]
    latentHeating = latentHeating[mask, :, :]
    latentHeating[latentHeating <= -9999.9] = np.nan

    # 设置CSH和SLh的高度.
    nlayer = 80
    dh = 0.25
    height = (np.arange(nlayer) + 0.5) * dh

    return {
        'Longitude': Longitude,
        'Latitude': Latitude,
        'latentHeating': latentHeating,
        'height': height
    }

def plot_precipitation():
    # 设置地图.
    crs = ccrs.PlateCarree()
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(projection=crs)
    ax.coastlines(resolution='50m', lw=0.5)
    ax.add_feature(LAND.with_scale('50m'))
    ax.add_feature(OCEAN.with_scale('50m'))
    fplt.set_extent_and_ticks(
        ax, extents,
        xticks=np.arange(-180, 181, 5),
        yticks=np.arange(-90, 91, 5),
        nx=1, ny=1
    )
    ax.set_extent(extents, crs)
    ax.tick_params(which='major', length=8, labelsize='large')
    ax.tick_params(which='minor', length=6)

    # 画出GMI降水.
    cf = ax.contourf(
        data_GMI['Longitude'],
        data_GMI['Latitude'],
        data_GMI['surfacePrecipitation'],
        levels=levels_Rr,
        cmap=cmap_Rr, norm=norm_Rr,
        extend='both', alpha=0.5,
        transform=crs
    )

    # 画出DPR降水
    cf = ax.contourf(
        data_DPR['Longitude'],
        data_DPR['Latitude'],
        data_DPR['precipRateNearSurface'],
        levels=levels_Rr,
        cmap=cmap_Rr, norm=norm_Rr,
        extend='both',
        transform=crs
    )

    cbar = fig.colorbar(cf, ax=ax, ticks=levels_Rr)
    cbar.set_label('Rain Rate (mm/hr)', fontsize='large')
    cbar.ax.tick_params(labelsize='large')

    # 画出横截面直线.
    for i, (start, end) in enumerate(start_ends):
        ax.plot(
            *zip(start, end),
            'o-', lw=1.5, c='k', ms=3,
            transform=crs
        )
        pos = get_label_pos(start, end)
        ax.text(
            pos[0], pos[1], str(i + 1),
            fontsize='large', weight='bold',
            ha='center', va='center',
            transform=crs
        )

    # 保存图片.
    ax.set_title(f'GMI and DPR Rain Rate on {date_str}', fontsize='x-large')
    filepath = dirpath_fig / 'precipitation.png'
    fig.savefig(str(filepath), dpi=300, bbox_inches='tight')
    plt.close(fig)

def plot_cross_section(i):
    # 计算横截面上的变量.
    cross_section = CrossSection(
        lon=data_DPR['Longitude'],
        lat=data_DPR['Latitude'],
        start=start_ends[i][0],
        end=start_ends[i][1]
    )
    elv1d = cross_section(data_DPR['elevation'])
    h01d = cross_section(data_DPR['heightZeroDeg'])
    Rr2d = cross_section(data_DPR['precipRate'])
    csh2d = cross_section(data_CSH['latentHeating'])
    slh2d = cross_section(data_SLH['latentHeating'])
    x, xticks, xticklabels = cross_section.get_xticks()

    fig, axes = plt.subplots(3, 1, figsize=(6, 8))
    fig.subplots_adjust(hspace=0.25)

    # 画出降水横截面.
    cmap_Rr_ = cmap_Rr.copy()
    cmap_Rr_.set_under('white')
    cf = axes[0].contourf(
        x, data_DPR['height'], Rr2d, levels_Rr,
        cmap=cmap_Rr_, norm=norm_Rr,
        extend='both'
    )
    cbar = fig.colorbar(
        cf, ax=axes[0],
        fraction=0.1, pad=0.25, aspect=40,
        orientation='horizontal',
        ticks=levels_Rr
    )
    cbar.set_label('Rain Rate (mm/hr)', fontsize='small')
    cbar.ax.tick_params(labelsize='small')

    # 画出CSH横截面.
    cf = axes[1].contourf(
        x, data_CSH['height'], csh2d, levels_LH,
        cmap=cmap_LH, norm=norm_LH,
        extend='both'
    )
    cbar = fig.colorbar(
        cf, ax=axes[1],
        fraction=0.1, pad=0.25, aspect=40,
        orientation='horizontal',
        ticks=levels_LH
    )
    cbar.set_label('Latent Heat (K/hr)', fontsize='small')
    cbar.ax.tick_params(labelsize='small')

    # 画出SLH横截面.
    cf = axes[2].contourf(
        x, data_SLH['height'], slh2d, levels_LH,
        cmap=cmap_LH, norm=norm_LH,
        extend='both'
    )
    cbar = fig.colorbar(
        cf, ax=axes[2],
        fraction=0.1, pad=0.25, aspect=40,
        orientation='horizontal',
        ticks=levels_LH
    )
    cbar.set_label('Latent Heat (K/hr)', fontsize='small')
    cbar.ax.tick_params(labelsize='small')

    # 画出地形和零度层高度.
    for ax in axes:
        ax.fill_between(x, elv1d, color='gray')
        ax.plot(x, h01d, 'k--', lw=1.2, label='0°C')
        ax.legend(
            loc='upper right',
            fontsize='small',
            fancybox=False,
            framealpha=1
        )

    # 设置坐标轴.
    for ax in axes:
        ax.set_ylim(0, 15)
        ax.set_ylabel('Height (km)', fontsize='small')
        ax.set_xticks(xticks)
        ax.set_xticklabels(xticklabels)
        ax.yaxis.set_major_locator(MultipleLocator(5))
        ax.yaxis.set_minor_locator(AutoMinorLocator(2))
        ax.xaxis.set_minor_locator(AutoMinorLocator(2))
        ax.tick_params(labelsize='small')

    # 添加标题.
    titles = [
        'DPR Rain Rate',
        'CSH Latent Heat',
        'SLH Latent Heat'
    ]
    for ax, title in zip(axes, titles):
        ax.text(
            0.02, 0.9, title,
            fontsize='small',
            ha='left', va='top',
            transform=ax.transAxes
        )

    # 保存图片.
    num = i + 1
    axes[0].set_title(f'Cross Section of Line {num}', fontsize='large')
    filepath = dirpath_fig / f'cross_section_{num:02}.png'
    fig.savefig(str(filepath), dpi=300, bbox_inches='tight')
    plt.close(fig)

# 目录路径.
dirpath_data = Path('../data')
dirpath_fig = Path('../fig')
if not dirpath_fig.exists():
    dirpath_fig.mkdir(parents=True)

# 文件路径.
filepath_GMI = next(dirpath_data.glob('*GPROF*'))
filepath_DPR = next(dirpath_data.glob('*DPR.V8*'))
filepath_CSH = next(dirpath_data.glob('*CSH*'))
filepath_SLH = next(dirpath_data.glob('*SLH*'))
date_str = get_date_str(filepath_DPR)

# 地图范围, 横截面起始点.
extents = [127, 147, 17, 37]
start_ends = [
    ((134.5, 26.5), (138.5, 26.5)),
    ((136.0, 22.0), (136.0, 29.0)),
    ((134.5, 22.8), (138.9, 30.7))
]

# 读取数据.
data_GMI = read_GMI_data(filepath_GMI)
data_DPR = read_DPR_data(filepath_DPR)
data_CSH = read_LH_data(filepath_CSH)
data_SLH = read_LH_data(filepath_SLH)

# 降水的cmap和norm.
levels_Rr = [0.1, 0.5, 1.0, 2.0, 3.0, 5.0, 10.0, 15.0, 20.0, 25, 30]
cmap_Rr = plt.cm.jet
cmap_Rr.set_under('lavender')
norm_Rr = BoundaryNorm(levels_Rr, cmap_Rr.N)

# 潜热的cmap和norm.
levels_LH = [-5, -2, -1, -0.5, -0.1, 0.1, 0.5, 1, 2, 5, 10, 20]
cmap_LH = cmaps.BlueWhiteOrangeRed
cmap_LH.set_under('white')
norm_LH = fplt.CenteredBoundaryNorm(levels_LH)

plot_precipitation()
for i in range(len(start_ends)):
    plot_cross_section(i)