import sqlite3
import sys
import re
from datetime import datetime
from pathlib import Path

import numpy as np
from osgeo import gdal, osr
from PyQt6 import QtWidgets
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QImage, QPixmap
from PyQt6.QtWidgets import (
    QFileDialog,
    QGraphicsScene,
    QMessageBox,
)

from rs import Ui_MainWindow


gdal.UseExceptions()


class RemoteSensingApp(QtWidgets.QMainWindow):
    """遥感影像读取、显示、指数计算和阈值提取程序。"""

    INDEX_DEFINITIONS = {
        # 指数名称: (分子第一个光谱角色, 分子第二个光谱角色)
        "NDVI": ("NIR", "RED"),
        "MNDWI": ("GREEN", "SWIR1"),
        "NDBI": ("SWIR1", "NIR"),
    }

    # 中心波长和允许偏差，单位均为微米。
    SPECTRAL_TARGETS = {
        "BLUE": (0.48, 0.12),
        "GREEN": (0.56, 0.12),
        "RED": (0.66, 0.12),
        "NIR": (0.84, 0.25),
        "SWIR1": (1.61, 0.35),
        "SWIR2": (2.20, 0.35),
    }

    def __init__(self):
        super().__init__()

        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)

        # 所有项目路径都以当前 app.py 所在目录为基准。
        # 整个项目文件夹移动到其他位置后仍可正常运行。
        self.project_directory = Path(__file__).resolve().parent
        self.data_directory = self.project_directory / "data"
        self.output_directory = self.project_directory / "outputs"
        self.data_directory.mkdir(exist_ok=True)
        self.output_directory.mkdir(exist_ok=True)

        # 当前数据集及其路径
        self.dataset = None
        self.source_path = None
        self.spectral_bands = {}

        # 当前指数结果
        self.index_array = None
        self.index_name = None

        # 当前可保存的输出结果
        self.output_array = None
        self.output_name = None
        self.output_is_binary = False

        # 当前阈值提取统计
        self.threshold_min = None
        self.threshold_max = None
        self.extracted_pixel_count = None
        self.extracted_area_km2 = None

        self.connect_signals()
        self.configure_graphics_views()

    def connect_signals(self):
        """连接界面控件与功能方法。"""
        self.ui.actionopen.triggered.connect(self.open_file)
        self.ui.actionsave.triggered.connect(self.save_result)
        self.ui.actionwarehousing.triggered.connect(self.save_to_database)

        self.ui.b_ok.clicked.connect(self.apply_band_combination)

        self.ui.veg.clicked.connect(
            lambda _checked=False: self.calculate_index("NDVI")
        )
        self.ui.water_2.clicked.connect(
            lambda _checked=False: self.calculate_index("MNDWI")
        )
        self.ui.nowater_2.clicked.connect(
            lambda _checked=False: self.calculate_index("NDBI")
        )

        self.ui.gt.clicked.connect(self.threshold_extraction)

    def configure_graphics_views(self):
        """配置影像窗口的平移和滚轮缩放。"""
        for view in (self.ui.graphicsView, self.ui.graphicsView_2):
            view.setDragMode(
                QtWidgets.QGraphicsView.DragMode.ScrollHandDrag
            )

        self.ui.graphicsView.wheelEvent = (
            lambda event: self.zoom_view(self.ui.graphicsView, event)
        )
        self.ui.graphicsView_2.wheelEvent = (
            lambda event: self.zoom_view(self.ui.graphicsView_2, event)
        )

    @staticmethod
    def zoom_view(view, event):
        """使用鼠标滚轮缩放指定影像窗口。"""
        if event.angleDelta().y() > 0:
            scale_factor = 1.15
        else:
            scale_factor = 1 / 1.15

        view.scale(scale_factor, scale_factor)

    def open_file(self, _checked=False):
        """选择并打开遥感影像。"""
        file_name, _ = QFileDialog.getOpenFileName(
            self,
            "选择遥感影像",
            str(self.data_directory),
            "遥感影像 (*.dat *.tif *.tiff *.img);;所有文件 (*)",
        )

        if not file_name:
            return

        try:
            dataset = gdal.Open(file_name, gdal.GA_ReadOnly)

            if dataset is None:
                raise RuntimeError("GDAL 无法打开该文件")

            self.dataset = dataset
            self.source_path = file_name
            self.spectral_bands = self.detect_spectral_bands(dataset)

            self.clear_result_state()
            self.ui.path.setText(self.get_portable_path(file_name))

            self.show_metadata(dataset, file_name)
            self.populate_band_selectors(dataset)

            default_rgb = self.get_default_rgb_bands(dataset)
            if default_rgb is not None:
                self.display_rgb(dataset, default_rgb)

            self.statusBar().showMessage("影像打开成功")

        except Exception as error:
            self.dataset = None
            self.source_path = None
            self.spectral_bands = {}

            QMessageBox.critical(
                self,
                "打开失败",
                f"无法打开遥感影像：\n{error}",
            )

    def get_portable_path(self, file_path):
        """项目内文件显示和入库时使用相对路径。"""
        resolved_path = Path(file_path).resolve()

        try:
            return str(resolved_path.relative_to(self.project_directory))
        except ValueError:
            # 仍允许用户打开项目目录之外的任意影像。
            return str(resolved_path)

    def clear_result_state(self):
        """打开新影像时清理上一个处理结果。"""
        self.index_array = None
        self.index_name = None

        self.output_array = None
        self.output_name = None
        self.output_is_binary = False

        self.threshold_min = None
        self.threshold_max = None
        self.extracted_pixel_count = None
        self.extracted_area_km2 = None

        self.ui.area.clear()

        empty_scene = QGraphicsScene(self.ui.graphicsView_2)
        self.ui.graphicsView_2.setScene(empty_scene)

    @staticmethod
    def normalize_wavelength_um(value, unit=""):
        """将常见的纳米或微米波长统一转换为微米。"""
        wavelength = float(value)
        normalized_unit = str(unit).strip().lower()

        if "nano" in normalized_unit or normalized_unit == "nm":
            return wavelength / 1000.0

        if (
            "micro" in normalized_unit
            or normalized_unit in {"um", "µm", "μm"}
        ):
            return wavelength

        # 缺少单位时采用保守的常见遥感波长量级判断。
        if wavelength > 100:
            return wavelength / 1000.0

        return wavelength

    def extract_wavelength_um(self, dataset, raster_band):
        """从波段描述或元数据中提取中心波长。"""
        description = (raster_band.GetDescription() or "").strip()
        description_match = re.search(
            r"([0-9]+(?:\.[0-9]+)?)\s*"
            r"(Micrometers?|Micrometers?|um|µm|μm|Nanometers?|nm)",
            description,
            flags=re.IGNORECASE,
        )

        if description_match:
            return self.normalize_wavelength_um(
                description_match.group(1),
                description_match.group(2),
            )

        metadata = {
            str(key).lower(): value
            for key, value in raster_band.GetMetadata().items()
        }

        wavelength_text = None
        for key in (
            "wavelength",
            "center_wavelength",
            "central_wavelength",
        ):
            if key in metadata:
                wavelength_text = metadata[key]
                break

        if wavelength_text is None:
            return None

        number_match = re.search(
            r"[0-9]+(?:\.[0-9]+)?",
            str(wavelength_text),
        )
        if not number_match:
            return None

        unit = (
            metadata.get("wavelength_units")
            or dataset.GetMetadataItem("wavelength_units")
            or ""
        )

        return self.normalize_wavelength_um(
            number_match.group(0),
            unit,
        )

    def format_band_label(self, dataset, band_number):
        """生成传感器无关的简洁波段名称。"""
        raster_band = dataset.GetRasterBand(band_number)
        description = (raster_band.GetDescription() or "").strip()
        wavelength = self.extract_wavelength_um(dataset, raster_band)

        band_match = re.search(
            r"\bBand\s*([0-9A-Za-z]+)\b",
            description,
            flags=re.IGNORECASE,
        )
        compact_match = re.search(
            r"\bB([0-9]{1,2}[A-Za-z]?)\b",
            description,
            flags=re.IGNORECASE,
        )

        if band_match:
            short_name = f"Band {band_match.group(1)}"
        elif compact_match:
            short_name = f"B{compact_match.group(1)}"
        elif description:
            short_name = description
        else:
            short_name = f"Band {band_number}"

        if wavelength is not None:
            return f"{short_name} - {wavelength:.3f} um"

        return short_name

    def detect_spectral_bands(self, dataset):
        """依据波长元数据识别常用光谱角色。"""
        wavelengths = {}

        for band_number in range(1, dataset.RasterCount + 1):
            raster_band = dataset.GetRasterBand(band_number)
            wavelength = self.extract_wavelength_um(dataset, raster_band)
            if wavelength is not None:
                wavelengths[band_number] = wavelength

        detected = {}
        for role, (target, tolerance) in self.SPECTRAL_TARGETS.items():
            if not wavelengths:
                break

            band_number, wavelength = min(
                wavelengths.items(),
                key=lambda item: abs(item[1] - target),
            )

            if abs(wavelength - target) <= tolerance:
                detected[role] = band_number

        return detected

    def get_default_rgb_bands(self, dataset):
        """优先使用已识别的真彩色波段，否则使用前三个波段。"""
        roles = ("RED", "GREEN", "BLUE")

        if all(role in self.spectral_bands for role in roles):
            return tuple(self.spectral_bands[role] for role in roles)

        if dataset.RasterCount >= 3:
            return (3, 2, 1)

        return None

    def show_metadata(self, dataset, file_name):
        """将影像元数据显示到右侧文本框。"""
        geo_transform = dataset.GetGeoTransform()
        projection = dataset.GetProjection()

        band_information = []

        for band_number in range(1, dataset.RasterCount + 1):
            raster_band = dataset.GetRasterBand(band_number)
            description = (raster_band.GetDescription() or "").strip()
            nodata_value = raster_band.GetNoDataValue()

            if not description:
                description = "无描述"

            wavelength = self.extract_wavelength_um(dataset, raster_band)
            wavelength_text = (
                f"，中心波长={wavelength:.3f} um"
                if wavelength is not None
                else ""
            )

            band_information.append(
                f"  文件波段 {band_number}：{description}，"
                f"NoData={nodata_value}{wavelength_text}"
            )

        detected_information = []
        for role in ("BLUE", "GREEN", "RED", "NIR", "SWIR1", "SWIR2"):
            if role in self.spectral_bands:
                detected_information.append(
                    f"  {role} -> 文件波段 {self.spectral_bands[role]}"
                )

        detected_text = (
            chr(10).join(detected_information)
            if detected_information
            else "  未从元数据中识别到可靠的光谱角色"
        )

        information = (
            f"文件：{self.get_portable_path(file_name)}\n\n"
            f"驱动：{dataset.GetDriver().ShortName}\n"
            f"宽度：{dataset.RasterXSize} 像元\n"
            f"高度：{dataset.RasterYSize} 像元\n"
            f"波段数：{dataset.RasterCount}\n"
            f"数据类型："
            f"{gdal.GetDataTypeName(dataset.GetRasterBand(1).DataType)}\n\n"
            f"波段信息：\n"
            f"{chr(10).join(band_information)}\n\n"
            f"自动识别：\n{detected_text}\n\n"
            f"地理变换：\n{geo_transform}\n\n"
            f"投影信息：\n{projection}"
        )

        self.ui.information.setPlainText(information)

    def populate_band_selectors(self, dataset):
        """根据实际数据集自动填写波段下拉框。"""
        combo_boxes = (self.ui.b1, self.ui.b2, self.ui.b3)

        for combo_box in combo_boxes:
            combo_box.clear()
            combo_box.setMinimumWidth(180)

        for band_number in range(1, dataset.RasterCount + 1):
            raster_band = dataset.GetRasterBand(band_number)
            description = (raster_band.GetDescription() or "").strip()
            display_name = self.format_band_label(dataset, band_number)
            tooltip = (
                f"文件内第 {band_number} 个波段"
                + (f"；原始描述：{description}" if description else "")
            )

            for combo_box in combo_boxes:
                combo_box.addItem(display_name, band_number)
                item_index = combo_box.count() - 1
                combo_box.setItemData(
                    item_index,
                    tooltip,
                    Qt.ItemDataRole.ToolTipRole,
                )

        default_rgb = self.get_default_rgb_bands(dataset)
        if default_rgb is not None:
            for combo_box, band_number in zip(combo_boxes, default_rgb):
                combo_box.setCurrentIndex(band_number - 1)

    def apply_band_combination(self):
        """按照下拉框选择重新生成 RGB 影像。"""
        if self.dataset is None:
            QMessageBox.warning(self, "提示", "请先打开遥感影像")
            return

        band_numbers = (
            self.ui.b1.currentData(),
            self.ui.b2.currentData(),
            self.ui.b3.currentData(),
        )

        if any(number is None for number in band_numbers):
            QMessageBox.warning(self, "提示", "请选择完整的 RGB 波段")
            return

        try:
            self.display_rgb(self.dataset, band_numbers)

            self.statusBar().showMessage(
                f"当前组合：R={band_numbers[0]}，"
                f"G={band_numbers[1]}，B={band_numbers[2]}"
            )

        except Exception as error:
            QMessageBox.critical(
                self,
                "波段组合失败",
                f"无法生成彩色合成影像：\n{error}",
            )

    @staticmethod
    def stretch_band(band_data, nodata_value):
        """使用2%—98%百分位拉伸，将单波段转换为8位数据。"""
        band_data = band_data.astype(np.float32)

        valid_mask = np.isfinite(band_data)

        if nodata_value is not None:
            valid_mask &= band_data != nodata_value

        valid_pixels = band_data[valid_mask]
        output = np.zeros(band_data.shape, dtype=np.uint8)

        if valid_pixels.size == 0:
            return output

        lower, upper = np.percentile(valid_pixels, (2, 98))

        if upper <= lower:
            return output

        stretched = (
            band_data[valid_mask] - lower
        ) / (upper - lower)

        output[valid_mask] = np.clip(
            stretched * 255,
            0,
            255,
        ).astype(np.uint8)

        return output

    def display_rgb(self, dataset, band_numbers=(3, 2, 1)):
        """按照指定的红、绿、蓝波段显示彩色合成影像。"""
        if dataset.RasterCount < 3:
            raise ValueError("影像少于三个波段，无法生成 RGB 影像")

        rgb_bands = []

        for band_number in band_numbers:
            raster_band = dataset.GetRasterBand(int(band_number))

            if raster_band is None:
                raise ValueError(f"不存在第 {band_number} 数据波段")

            band_data = raster_band.ReadAsArray()

            if band_data is None:
                raise RuntimeError(f"无法读取第 {band_number} 波段")

            display_band = self.stretch_band(
                band_data,
                raster_band.GetNoDataValue(),
            )

            rgb_bands.append(display_band)

        rgb_array = np.dstack(rgb_bands)
        self.show_rgb_array(self.ui.graphicsView, rgb_array)

    def show_rgb_array(self, graphics_view, rgb_array):
        """在指定 QGraphicsView 中显示三通道数组。"""
        rgb_array = np.ascontiguousarray(rgb_array, dtype=np.uint8)

        height, width, channel_count = rgb_array.shape
        bytes_per_line = width * channel_count

        image = QImage(
            rgb_array.data,
            width,
            height,
            bytes_per_line,
            QImage.Format.Format_RGB888,
        ).copy()

        pixmap = QPixmap.fromImage(image)

        scene = QGraphicsScene(graphics_view)
        scene.addPixmap(pixmap)

        graphics_view.setScene(scene)
        graphics_view.fitInView(
            scene.itemsBoundingRect(),
            Qt.AspectRatioMode.KeepAspectRatio,
        )

    def calculate_index(self, index_name):
        """计算 NDVI、MNDWI 或 NDBI。"""
        if self.dataset is None:
            QMessageBox.warning(self, "提示", "请先打开遥感影像")
            return

        first_role, second_role = (
            self.INDEX_DEFINITIONS[index_name]
        )

        missing_roles = [
            role
            for role in (first_role, second_role)
            if role not in self.spectral_bands
        ]

        if missing_roles:
            QMessageBox.warning(
                self,
                "无法识别指数波段",
                f"{index_name} 需要识别 "
                f"{first_role} 和 {second_role} 波段。\n"
                f"当前数据缺少可识别的：{', '.join(missing_roles)}。\n\n"
                "请确认影像波段描述或元数据中包含中心波长。",
            )
            return

        first_band_number = self.spectral_bands[first_role]
        second_band_number = self.spectral_bands[second_role]

        if first_band_number == second_band_number:
            QMessageBox.warning(
                self,
                "波段识别冲突",
                f"{first_role} 和 {second_role} 被识别为同一波段，"
                "无法可靠计算指数。",
            )
            return

        try:
            first_band = self.dataset.GetRasterBand(
                first_band_number
            )
            second_band = self.dataset.GetRasterBand(
                second_band_number
            )

            first_array = first_band.ReadAsArray().astype(np.float32)
            second_array = second_band.ReadAsArray().astype(np.float32)

            valid_mask = (
                np.isfinite(first_array)
                & np.isfinite(second_array)
            )

            first_nodata = first_band.GetNoDataValue()
            second_nodata = second_band.GetNoDataValue()

            if first_nodata is not None:
                valid_mask &= first_array != first_nodata

            if second_nodata is not None:
                valid_mask &= second_array != second_nodata

            numerator = first_array - second_array
            denominator = first_array + second_array

            valid_mask &= denominator != 0

            index_array = np.full(
                first_array.shape,
                np.nan,
                dtype=np.float32,
            )

            np.divide(
                numerator,
                denominator,
                out=index_array,
                where=valid_mask,
            )

            # 比值型指数理论范围通常为 -1 到 1
            index_array = np.clip(index_array, -1.0, 1.0)

            self.index_array = index_array
            self.index_name = index_name

            self.output_array = index_array
            self.output_name = index_name
            self.output_is_binary = False

            self.threshold_min = None
            self.threshold_max = None
            self.extracted_pixel_count = None
            self.extracted_area_km2 = None
            self.ui.area.clear()

            self.display_index(index_array)

            self.statusBar().showMessage(
                f"{index_name} 计算完成："
                f"{first_role}=文件波段{first_band_number}，"
                f"{second_role}=文件波段{second_band_number}"
            )

        except Exception as error:
            QMessageBox.critical(
                self,
                "指数计算失败",
                f"无法计算 {index_name}：\n{error}",
            )

    def display_index(self, index_array):
        """将 -1 到 1 的指数显示为灰度影像。"""
        valid_mask = np.isfinite(index_array)

        display_array = np.zeros(
            index_array.shape,
            dtype=np.uint8,
        )

        display_array[valid_mask] = np.clip(
            (index_array[valid_mask] + 1.0) * 127.5,
            0,
            255,
        ).astype(np.uint8)

        rgb_array = np.dstack(
            (display_array, display_array, display_array)
        )

        self.show_rgb_array(
            self.ui.graphicsView_2,
            rgb_array,
        )

    def threshold_extraction(self):
        """按照输入的上下限提取指数范围并计算面积。"""
        if self.index_array is None:
            QMessageBox.warning(
                self,
                "提示",
                "请先计算 NDVI、MNDWI 或 NDBI",
            )
            return

        minimum_text = self.ui.gt1.text().strip()
        maximum_text = self.ui.gt2.text().strip()

        try:
            minimum = (
                float(minimum_text)
                if minimum_text
                else -1.0
            )
            maximum = (
                float(maximum_text)
                if maximum_text
                else 1.0
            )
        except ValueError:
            QMessageBox.warning(
                self,
                "阈值错误",
                "最小阈值和最大阈值必须是数字",
            )
            return

        if minimum > maximum:
            QMessageBox.warning(
                self,
                "阈值错误",
                "最小阈值不能大于最大阈值",
            )
            return

        valid_mask = np.isfinite(self.index_array)

        extraction_mask = (
            valid_mask
            & (self.index_array >= minimum)
            & (self.index_array <= maximum)
        )

        binary_array = np.where(
            extraction_mask,
            255,
            0,
        ).astype(np.uint8)

        extracted_pixel_count = int(
            np.count_nonzero(extraction_mask)
        )

        try:
            pixel_area_m2 = self.calculate_pixel_area_m2()
        except Exception as error:
            QMessageBox.critical(
                self,
                "面积计算失败",
                str(error),
            )
            return

        extracted_area_km2 = (
            extracted_pixel_count
            * pixel_area_m2
            / 1_000_000
        )

        self.threshold_min = minimum
        self.threshold_max = maximum
        self.extracted_pixel_count = extracted_pixel_count
        self.extracted_area_km2 = extracted_area_km2

        self.output_array = binary_array
        self.output_name = f"{self.index_name}_threshold"
        self.output_is_binary = True

        self.ui.area.setText(
            f"{extracted_area_km2:.4f}"
        )

        rgb_array = np.dstack(
            (binary_array, binary_array, binary_array)
        )

        self.show_rgb_array(
            self.ui.graphicsView_2,
            rgb_array,
        )

        self.statusBar().showMessage(
            f"{self.index_name} 阈值提取完成："
            f"{extracted_pixel_count} 个像元，"
            f"{extracted_area_km2:.4f} km²"
        )

    def calculate_pixel_area_m2(self):
        """根据仿射变换和投影单位计算单像元面积。"""
        if self.dataset is None:
            raise RuntimeError("当前没有打开的数据集")

        geo_transform = self.dataset.GetGeoTransform()
        projection = self.dataset.GetProjection()

        if not projection:
            raise ValueError(
                "影像没有投影信息，不能直接计算面积"
            )

        spatial_reference = osr.SpatialReference()

        if spatial_reference.ImportFromWkt(projection) != 0:
            raise ValueError("无法解析影像投影信息")

        if not spatial_reference.IsProjected():
            raise ValueError(
                "当前影像使用地理坐标系，"
                "不能直接用经纬度像元计算面积"
            )

        # 仿射矩阵行列式，可兼容旋转栅格
        pixel_area_in_crs_units = abs(
            geo_transform[1] * geo_transform[5]
            - geo_transform[2] * geo_transform[4]
        )

        # GetLinearUnits 返回“一个坐标单位对应多少米”
        linear_unit_to_metre = (
            spatial_reference.GetLinearUnits()
        )

        pixel_area_m2 = (
            pixel_area_in_crs_units
            * linear_unit_to_metre**2
        )

        return pixel_area_m2

    def save_result(self, _checked=False):
        """保存当前指数或阈值结果并继承空间参考。"""
        if self.dataset is None or self.output_array is None:
            QMessageBox.warning(
                self,
                "提示",
                "当前没有可以保存的处理结果",
            )
            return

        default_name = (
            f"{Path(self.source_path).stem}_"
            f"{self.output_name}.tif"
        )
        default_path = self.output_directory / default_name

        file_name, selected_filter = QFileDialog.getSaveFileName(
            self,
            "保存处理结果",
            str(default_path),
            "GeoTIFF (*.tif *.tiff);;ENVI DAT (*.dat)",
        )

        if not file_name:
            return

        output_path = Path(file_name)

        if selected_filter.startswith("ENVI"):
            driver_name = "ENVI"

            if output_path.suffix.lower() != ".dat":
                output_path = output_path.with_suffix(".dat")
        else:
            driver_name = "GTiff"

            if output_path.suffix.lower() not in (
                ".tif",
                ".tiff",
            ):
                output_path = output_path.with_suffix(".tif")

        try:
            height, width = self.output_array.shape

            if self.output_is_binary:
                gdal_data_type = gdal.GDT_Byte
                write_array = self.output_array.astype(np.uint8)
                nodata_value = None
            else:
                gdal_data_type = gdal.GDT_Float32
                nodata_value = -9999.0

                write_array = np.where(
                    np.isfinite(self.output_array),
                    self.output_array,
                    nodata_value,
                ).astype(np.float32)

            driver = gdal.GetDriverByName(driver_name)

            if driver is None:
                raise RuntimeError(
                    f"无法获取 {driver_name} 驱动"
                )

            creation_options = []

            if driver_name == "GTiff":
                creation_options = [
                    "COMPRESS=LZW",
                    "TILED=YES",
                ]

            output_dataset = driver.Create(
                str(output_path),
                width,
                height,
                1,
                gdal_data_type,
                options=creation_options,
            )

            if output_dataset is None:
                raise RuntimeError("无法创建输出文件")

            output_dataset.SetGeoTransform(
                self.dataset.GetGeoTransform()
            )
            output_dataset.SetProjection(
                self.dataset.GetProjection()
            )

            output_band = output_dataset.GetRasterBand(1)

            if nodata_value is not None:
                output_band.SetNoDataValue(nodata_value)

            output_band.WriteArray(write_array)
            output_band.FlushCache()
            output_dataset.FlushCache()

            output_dataset = None

            QMessageBox.information(
                self,
                "保存成功",
                f"结果已保存到：\n{output_path}",
            )

            self.statusBar().showMessage(
                f"结果已保存：{output_path.name}"
            )

        except Exception as error:
            QMessageBox.critical(
                self,
                "保存失败",
                f"无法保存处理结果：\n{error}",
            )

    def save_to_database(self, _checked=False):
        """将阈值、像元数量和面积写入 SQLite。"""
        if (
            self.index_name is None
            or self.extracted_area_km2 is None
            or self.extracted_pixel_count is None
        ):
            QMessageBox.warning(
                self,
                "提示",
                "请先完成指数计算和阈值提取",
            )
            return

        database_path = self.output_directory / "area_results.db"

        try:
            with sqlite3.connect(database_path) as connection:
                cursor = connection.cursor()

                cursor.execute(
                    """
                    CREATE TABLE IF NOT EXISTS extraction_results (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        source_file TEXT NOT NULL,
                        index_name TEXT NOT NULL,
                        minimum_threshold REAL NOT NULL,
                        maximum_threshold REAL NOT NULL,
                        pixel_count INTEGER NOT NULL,
                        area_km2 REAL NOT NULL,
                        created_at TEXT NOT NULL
                    )
                    """
                )

                cursor.execute(
                    """
                    INSERT INTO extraction_results (
                        source_file,
                        index_name,
                        minimum_threshold,
                        maximum_threshold,
                        pixel_count,
                        area_km2,
                        created_at
                    )
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        self.get_portable_path(self.source_path),
                        self.index_name,
                        self.threshold_min,
                        self.threshold_max,
                        self.extracted_pixel_count,
                        self.extracted_area_km2,
                        datetime.now().isoformat(
                            timespec="seconds"
                        ),
                    ),
                )

                connection.commit()

            QMessageBox.information(
                self,
                "入库成功",
                f"统计结果已写入：\n{database_path}",
            )

            self.statusBar().showMessage(
                "面积统计结果已写入 SQLite"
            )

        except Exception as error:
            QMessageBox.critical(
                self,
                "入库失败",
                f"无法写入 SQLite：\n{error}",
            )


if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)

    window = RemoteSensingApp()
    window.show()

    sys.exit(app.exec())
