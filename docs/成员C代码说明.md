# 成员 C：融合定位 + 区域增长/分水岭字符分割代码说明

成员 C 的完整代码位于：

```text
src/member_c_region_watershed.py
```

该脚本遵守项目统一预测 CSV 接口，输出字段可直接交给 `src/evaluate.py` 和
`src/visualize_single.py` 使用。

## 1. 方法名称

默认方法名：

```text
member_c_fusion_region_watershed_v1
```

默认输出：

```text
results/member_c_fusion_region_watershed_v1_predictions.csv
```

## 2. 运行模式

脚本支持三种车牌框来源：

| 参数 | 含义 | 用途 |
|---|---|---|
| `--plate-source gt` | 使用人工标注车牌框 | 单独评估成员 C 的字符分割能力 |
| `--plate-source pred` | 使用其他脚本输出的车牌框 | 交叉组合实验 |
| `--plate-source auto` | 使用成员 C 自己的融合定位方法 | 完整端到端实验 |

## 3. 端到端流程

成员 C 的端到端流程为：

```text
原图
  -> 边缘候选框
  -> HSV 颜色候选框
  -> 候选框融合与评分
  -> 车牌 ROI
  -> 阈值种子
  -> 区域增长
  -> 标记控制分水岭
  -> 字符框输出
```

## 4. 车牌定位部分

车牌定位函数：

```text
detect_plate()
```

主要步骤：

1. 灰度化并使用 CLAHE 增强局部对比度。
2. 用 Sobel-x 提取竖直笔画和车牌边框的横向梯度。
3. 使用 Otsu 阈值得到边缘二值图。
4. 用横向闭运算把车牌内部密集边缘连接成候选区域。
5. 在 HSV 空间中提取蓝牌、黄牌和高亮白牌候选区域。
6. 将边缘候选和颜色候选做 IoU 融合。
7. 对候选框按长宽比、边缘密度、颜色占比、矩形度和位置先验打分。

评分形式对应实验计划中的融合候选框思想：

```text
score = 0.30 * aspect_score
      + 0.25 * edge_density_score
      + 0.20 * color_ratio_score
      + 0.15 * rectangularity_score
      + 0.10 * position_score
```

其中融合候选框会获得少量额外加分，因为它同时被边缘和颜色线索支持。

## 5. 字符分割部分

字符分割主函数：

```text
run_char_segmentation()
segment_characters()
```

主要步骤：

1. 根据车牌框裁剪 ROI，并将 ROI 归一化到固定高度。
2. 使用 CLAHE 和轻微高斯滤波提高二值化稳定性。
3. 生成四类初始种子：
   - Otsu 亮字符
   - Otsu 暗字符
   - 自适应阈值亮字符
   - 自适应阈值暗字符
4. 对每类种子执行区域增长：
   - 先腐蚀得到更可靠的字符核心种子。
   - 以种子灰度中位数作为参考。
   - 只允许灰度差不超过 `grow_similarity` 的邻域像素加入。
   - 设置最大前景比例，防止区域无限扩散。
5. 对区域增长后的前景图执行标记控制分水岭：
   - 优先使用距离变换生成字符内部 marker。
   - marker 不足时使用 7 个字符槽位生成 slot marker。
   - 在梯度图上运行 `cv2.watershed()`。
6. 从分水岭标签或连通域中提取候选字符框。
7. 若候选框数量不是 7，则用投影曲线进行合并、切分或退回 7 等分先验。
8. 对不同二值模式、区域增长结果和分水岭结果打分，选择最佳结果。

## 6. 输出字段

脚本输出的关键字段包括：

| 字段 | 说明 |
|---|---|
| `image_name` | 图像文件名 |
| `method` | 方法名 |
| `plate_bbox_pred` | 车牌预测框 JSON |
| `char_bboxes_pred` | 7 个字符预测框 JSON |
| `params` | 参数 JSON |
| `runtime_ms` | 单张图耗时 |
| `status` | `success`、`plate_not_found`、`char_failed` 等 |
| `failure_reason` | 失败原因 |
| `binary_path` | 可选，区域增长后的字符前景图 |
| `foreground_path` | 可选，分水岭 marker 可视化图 |

所有框均为原图坐标 `[x, y, w, h]`。

## 7. 以后有 Python 环境时的运行命令

只评估成员 C 的字符分割：

```powershell
python src\member_c_region_watershed.py --plate-source gt --split all --out results\member_c_region_gt_predictions.csv
```

完整端到端运行：

```powershell
python src\member_c_region_watershed.py --plate-source auto --split all --out results\member_c_fusion_region_watershed_v1_predictions.csv
```

使用其他成员的车牌定位框做交叉实验：

```powershell
python src\member_c_region_watershed.py --plate-source pred --plate-pred results\edge_morph_plate_rect_heavy_v2_predictions.csv --split all --out results\member_c_region_with_edge_plate_predictions.csv
```

保存调试图：

```powershell
python src\member_c_region_watershed.py --plate-source auto --save-debug --out results\member_c_fusion_region_watershed_v1_predictions.csv
```

评估：

```powershell
python src\evaluate.py --gt annotations\plate_char_annotations.csv --pred results\member_c_fusion_region_watershed_v1_predictions.csv --out-dir results\eval_member_c_fusion_region_watershed_v1
```

可视化：

```powershell
python src\visualize_single.py --gt annotations\plate_char_annotations.csv --pred results\member_c_fusion_region_watershed_v1_predictions.csv --image-dir dataset --out-dir outputs\single_vis_member_c_fusion_region_watershed_v1 --metrics results\eval_member_c_fusion_region_watershed_v1\per_image_metrics.csv
```

## 8. 当前验证状态

当前电脑没有可用 Python 环境，因此本脚本未进行实际运行验证。代码已按现有项目脚本的输入输出协议编写，后续只需要在配置好 Python、OpenCV、NumPy、Pandas 后运行上面的命令即可生成预测、评估和可视化结果。
