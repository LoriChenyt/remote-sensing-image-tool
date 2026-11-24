# 测试数据说明

本地项目目录包含用于复现功能的 Landsat TM 测试影像；大型影像默认被 `.gitignore` 排除，不随代码仓库发布。

本项目当前使用 Landsat TM ENVI 数据进行测试。本目录输入文件包括：

```text
TM.dat
TM.hdr
TM.dat.aux.xml   可选
TM.dat.ovr       显示金字塔，可重新生成
```

已验证数据特征：

- 5292 × 2840 像元
- 6 个数据波段，对应 Landsat TM B1、B2、B3、B4、B5、B7
- EPSG:32650
- 30 m × 30 m 像元
- NoData 为 0

程序打开文件对话框时默认进入本目录。请在公开数据前确认数据来源、许可和再分发条件。`.ovr` 金字塔文件可以由 GIS 软件重新生成，无需纳入仓库。

