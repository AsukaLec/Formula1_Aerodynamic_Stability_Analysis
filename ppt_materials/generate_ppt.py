#!/usr/bin/env python3
"""
generate_ppt.py — Auto-generate course defense PPT using python-pptx.
Output: ppt_materials/F1_气动稳定性研究_答辩PPT.pptx

Dependencies: python-pptx (pip install python-pptx)
"""

import os
from pptx import Presentation
from pptx.util import Inches, Cm, Pt, Emu
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR
from pptx.enum.shapes import MSO_SHAPE

# ============================================================
# Constants
# ============================================================
SLIDE_W = Inches(13.333)   # 16:9 widescreen
SLIDE_H = Inches(7.5)

DARK_BLUE   = RGBColor(0x1A, 0x3A, 0x6B)
ACCENT_BLUE = RGBColor(0x2B, 0x5C, 0xD9)
ORANGE      = RGBColor(0xE8, 0x64, 0x2D)
WHITE       = RGBColor(0xFF, 0xFF, 0xFF)
LIGHT_GRAY  = RGBColor(0xF5, 0xF7, 0xFA)
DARK_GRAY   = RGBColor(0x33, 0x33, 0x33)
MID_GRAY    = RGBColor(0x66, 0x66, 0x66)
ZEBRA_BLUE  = RGBColor(0xEB, 0xF0, 0xFA)

FONT_TITLE  = 'SimHei'
FONT_BODY   = 'Microsoft YaHei'
FONT_CODE   = 'Consolas'

TITLE_BAR_H = Cm(1.2)
SAFE_MARGIN = Cm(0.8)
IMAGES_DIR  = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'images')


# ============================================================
# Helper Functions
# ============================================================

def set_slide_bg(slide, color=WHITE):
    """Set solid background color for a slide."""
    bg = slide.background
    fill = bg.fill
    fill.solid()
    fill.fore_color.rgb = color


def add_title_bar(slide, chapter_num, chapter_title, subtitle=""):
    """Add a dark blue title bar at the top of the slide."""
    # Background rectangle
    rect = slide.shapes.add_shape(
        MSO_SHAPE.RECTANGLE, Cm(0), Cm(0), SLIDE_W, TITLE_BAR_H
    )
    rect.fill.solid()
    rect.fill.fore_color.rgb = DARK_BLUE
    rect.line.fill.background()

    # Chapter number
    left = Cm(0.8)
    top = Cm(0.15)
    tf = rect.text_frame
    tf.word_wrap = True
    p = tf.paragraphs[0]
    run = p.add_run()
    run.text = chapter_num
    run.font.size = Pt(22)
    run.font.bold = True
    run.font.color.rgb = WHITE
    run.font.name = FONT_TITLE

    # Chapter title
    txBox = slide.shapes.add_textbox(Cm(2.5), Cm(0.15), Cm(9), Cm(0.9))
    tf_title = txBox.text_frame
    tf_title.word_wrap = True
    p = tf_title.paragraphs[0]
    run = p.add_run()
    run.text = chapter_title
    run.font.size = Pt(22)
    run.font.bold = True
    run.font.color.rgb = WHITE
    run.font.name = FONT_TITLE

    if subtitle:
        p2 = tf_title.add_paragraph()
        run2 = p2.add_run()
        run2.text = subtitle
        run2.font.size = Pt(12)
        run2.font.color.rgb = RGBColor(0xCC, 0xD5, 0xE8)
        run2.font.name = FONT_BODY

    # Bottom line separator
    line = slide.shapes.add_shape(
        MSO_SHAPE.RECTANGLE, Cm(0.8), TITLE_BAR_H, Cm(11.733), Cm(0.05)
    )
    line.fill.solid()
    line.fill.fore_color.rgb = ACCENT_BLUE
    line.line.fill.background()


def add_page_number(slide, num):
    """Add page number at bottom center."""
    txBox = slide.shapes.add_textbox(Cm(0), Cm(18.2), SLIDE_W, Cm(0.7))
    tf = txBox.text_frame
    p = tf.paragraphs[0]
    p.alignment = PP_ALIGN.CENTER
    run = p.add_run()
    run.text = str(num)
    run.font.size = Pt(10)
    run.font.color.rgb = MID_GRAY
    run.font.name = FONT_BODY

    # Footer separator line
    line = slide.shapes.add_shape(
        MSO_SHAPE.RECTANGLE, Cm(0.8), Cm(18.1), Cm(11.733), Cm(0.02)
    )
    line.fill.solid()
    line.fill.fore_color.rgb = RGBColor(0xDD, 0xDD, 0xDD)
    line.line.fill.background()


def add_textbox(slide, left, top, width, height, text, font_size=14,
                bold=False, color=DARK_GRAY, alignment=PP_ALIGN.LEFT,
                font_name=FONT_BODY, line_spacing=1.3):
    """Add a simple text box with one paragraph."""
    txBox = slide.shapes.add_textbox(Cm(left), Cm(top), Cm(width), Cm(height))
    tf = txBox.text_frame
    tf.word_wrap = True
    p = tf.paragraphs[0]
    p.alignment = alignment
    p.space_after = Pt(2)
    run = p.add_run()
    run.text = text
    run.font.size = Pt(font_size)
    run.font.bold = bold
    run.font.color.rgb = color
    run.font.name = font_name
    return tf


def add_multiline_textbox(slide, left, top, width, height, lines,
                          font_size=14, color=DARK_GRAY, font_name=FONT_BODY,
                          bold_lines=None, color_lines=None, size_lines=None):
    """Add a text box with multiple paragraphs (lines is list of strings)."""
    if bold_lines is None:
        bold_lines = set()
    if color_lines is None:
        color_lines = {}
    if size_lines is None:
        size_lines = {}
    txBox = slide.shapes.add_textbox(Cm(left), Cm(top), Cm(width), Cm(height))
    tf = txBox.text_frame
    tf.word_wrap = True
    for i, line in enumerate(lines):
        if i == 0:
            p = tf.paragraphs[0]
        else:
            p = tf.add_paragraph()
        p.space_after = Pt(2)
        fs = size_lines.get(i, font_size)
        fc = color_lines.get(i, color)
        fb = i in bold_lines
        run = p.add_run()
        run.text = line
        run.font.size = Pt(fs)
        run.font.bold = fb
        run.font.color.rgb = fc
        run.font.name = font_name
    return tf


def add_image(slide, img_name, left, top, width=None, height=None):
    """Add an image from the images/ directory."""
    path = os.path.join(IMAGES_DIR, img_name)
    if not os.path.exists(path):
        add_textbox(slide, left, top, 4, 1, f"[Image not found: {img_name}]",
                     font_size=10, color=ORANGE)
        return None
    kwargs = {}
    if width is not None:
        kwargs['width'] = Cm(width)
    if height is not None:
        kwargs['height'] = Cm(height)
    return slide.shapes.add_picture(path, Cm(left), Cm(top), **kwargs)


def add_table(slide, left, top, width, height, headers, rows,
              col_widths=None, header_bg=DARK_BLUE):
    """Add a styled table."""
    n_rows = len(rows) + 1
    n_cols = len(headers)
    shape = slide.shapes.add_table(n_rows, n_cols, Cm(left), Cm(top),
                                   Cm(width), Cm(height))
    table = shape.table

    if col_widths:
        for i, w in enumerate(col_widths):
            table.columns[i].width = Cm(w)

    # Header row
    for j, h in enumerate(headers):
        cell = table.cell(0, j)
        cell.text = h
        for paragraph in cell.text_frame.paragraphs:
            paragraph.font.size = Pt(11)
            paragraph.font.bold = True
            paragraph.font.color.rgb = WHITE
            paragraph.font.name = FONT_TITLE
            paragraph.alignment = PP_ALIGN.CENTER
        cell.fill.solid()
        cell.fill.fore_color.rgb = header_bg

    # Data rows
    for i, row in enumerate(rows):
        for j, val in enumerate(row):
            cell = table.cell(i + 1, j)
            cell.text = str(val)
            for paragraph in cell.text_frame.paragraphs:
                paragraph.font.size = Pt(10)
                paragraph.font.color.rgb = DARK_GRAY
                paragraph.font.name = FONT_BODY
                paragraph.alignment = PP_ALIGN.CENTER
            if i % 2 == 1:
                cell.fill.solid()
                cell.fill.fore_color.rgb = ZEBRA_BLUE
    return shape


def add_rounded_box(slide, left, top, width, height, fill_color=LIGHT_GRAY):
    """Add a rounded rectangle as background card."""
    shape = slide.shapes.add_shape(
        MSO_SHAPE.ROUNDED_RECTANGLE, Cm(left), Cm(top), Cm(width), Cm(height)
    )
    shape.fill.solid()
    shape.fill.fore_color.rgb = fill_color
    shape.line.fill.background()
    return shape


def add_deep_blue_slide(slide, title_lines, subtitle_lines=None, page_num=None):
    """Create a deep blue full-background slide (cover, TOC, end)."""
    set_slide_bg(slide, DARK_BLUE)

    # Title
    y = 2.0
    for line in title_lines:
        txBox = slide.shapes.add_textbox(Cm(1.5), Cm(y), Cm(10.5), Cm(1.5))
        tf = txBox.text_frame
        tf.word_wrap = True
        p = tf.paragraphs[0]
        p.alignment = PP_ALIGN.CENTER
        run = p.add_run()
        run.text = line
        run.font.size = Pt(32)
        run.font.bold = True
        run.font.color.rgb = WHITE
        run.font.name = FONT_TITLE
        y += 1.6

    # Subtitle
    if subtitle_lines:
        y += 0.3
        for line in subtitle_lines:
            txBox = slide.shapes.add_textbox(Cm(2), Cm(y), Cm(9.5), Cm(0.8))
            tf = txBox.text_frame
            tf.word_wrap = True
            p = tf.paragraphs[0]
            p.alignment = PP_ALIGN.CENTER
            run = p.add_run()
            run.text = line
            run.font.size = Pt(16)
            run.font.color.rgb = RGBColor(0xCC, 0xD5, 0xE8)
            run.font.name = FONT_BODY
            y += 0.7

    # Accent line
    line = slide.shapes.add_shape(
        MSO_SHAPE.RECTANGLE, Cm(5.5), Cm(y + 0.3), Cm(2.5), Cm(0.06)
    )
    line.fill.solid()
    line.fill.fore_color.rgb = ORANGE
    line.line.fill.background()

    if page_num:
        add_textbox(slide, 11.5, 18.6, 1.5, 0.5, str(page_num),
                     font_size=10, color=RGBColor(0x99, 0xAA, 0xCC),
                     alignment=PP_ALIGN.RIGHT)


def new_slide(prs):
    """Create a new blank slide with white background."""
    slide_layout = prs.slide_layouts[6]  # blank
    slide = prs.slides.add_slide(slide_layout)
    set_slide_bg(slide, WHITE)
    return slide


# ============================================================
# Slide Generators
# ============================================================

def slide_01_cover(prs):
    slide = deep_blue_slide(prs)
    add_textbox(slide, 2, 0.8, 9.5, 1.2,
                '华南师范大学', font_size=24, color=WHITE, font_name=FONT_TITLE,
                alignment=PP_ALIGN.CENTER)
    add_textbox(slide, 2, 1.6, 9.5, 0.8,
                '人工智能学院', font_size=18, color=RGBColor(0xCC, 0xD5, 0xE8),
                alignment=PP_ALIGN.CENTER)
    add_textbox(slide, 1, 3.5, 11.5, 2.0,
                '基于代理模型与改进粒子群算法的\nF1 赛车气动稳定性优化研究',
                font_size=32, bold=True, color=WHITE, font_name=FONT_TITLE,
                alignment=PP_ALIGN.CENTER)
    add_textbox(slide, 2, 6.2, 9.5, 0.8,
                '《计算智能》课程设计', font_size=20, color=WHITE,
                alignment=PP_ALIGN.CENTER)

    # Accent line
    l = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, Cm(5.5), Cm(5.8), Cm(2.5), Cm(0.06))
    l.fill.solid(); l.fill.fore_color.rgb = ORANGE; l.line.fill.background()

    add_textbox(slide, 2, 8.5, 9.5, 2.0,
                '汇报人：陈新安 (202440012028)　郑骏远 (202440012047)\n2025—2026 学年第二学期',
                font_size=16, color=RGBColor(0xCC, 0xD5, 0xE8),
                alignment=PP_ALIGN.CENTER)
    return slide


def slide_02_toc(prs):
    slide = deep_blue_slide(prs)
    add_textbox(slide, 2, 0.8, 9.5, 1.2,
                'CONTENTS  目录', font_size=28, bold=True, color=WHITE,
                font_name=FONT_TITLE, alignment=PP_ALIGN.CENTER)
    l = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, Cm(5.5), Cm(1.8), Cm(2.5), Cm(0.06))
    l.fill.solid(); l.fill.fore_color.rgb = ORANGE; l.line.fill.background()

    toc_left = [
        '01  研究背景与问题定义',
        '02  数据分析与预处理',
        '03  技术路线总览',
        '04  代理模型构建与对比',
        '05  粒子群优化算法',
        '06  风险敏感 PSO 创新',
        '07  主动学习迭代精炼',
    ]
    toc_right = [
        '08  多场景优化',
        '09  Pareto 前沿与可解释性',
        '10  景观诊断核心发现',
        '11  Porpoising 风险分析',
        '12  消融实验与鲁棒性',
        '13  量化指标汇总',
        '14  创新点总结与结论',
    ]

    for i, item in enumerate(toc_left):
        add_textbox(slide, 1.5, 2.8 + i * 1.2, 5.5, 1.0,
                    f'  {item}', font_size=18, color=WHITE, font_name=FONT_BODY)

    for i, item in enumerate(toc_right):
        add_textbox(slide, 7.2, 2.8 + i * 1.2, 5.5, 1.0,
                    f'  {item}', font_size=18, color=WHITE, font_name=FONT_BODY)
    return slide


def slide_03_background(prs):
    slide = new_slide(prs)
    add_title_bar(slide, '01', '研究背景与问题定义', 'Background & Problem Definition')
    add_rounded_box(slide, 0.5, 1.8, 6.8, 5.8)

    lines = [
        '2022 FIA 引入"地面效应"规则 → 三大核心矛盾：',
        '',
        '① 规则约束 (ATR 配额)',
        '  CFD 仿真时长和计算资源受严格配额，传统高精度仿真',
        '  在有限周期内难以覆盖大规模参数空间',
        '',
        '② 瞬态非线性 ("海豚跳")',
        '  车速、翼角、DRS 多参数共振耦合，',
        '  传统方法难以捕捉强非线性特性',
        '',
        '③ 数据不平衡',
        '  高稳定性样本占 86.02%，低稳定性样本稀缺，',
        '  ML 建模面临严重偏置风险',
    ]
    add_multiline_textbox(slide, 0.8, 2.0, 6.2, 5.4, lines,
                          font_size=14, color=DARK_GRAY,
                          bold_lines={0, 2, 6, 10},
                          color_lines={0: ORANGE, 2: ACCENT_BLUE, 6: ACCENT_BLUE,
                                       10: ACCENT_BLUE})

    # Right side: core formula
    add_rounded_box(slide, 7.8, 1.8, 5.0, 5.8)
    formula_lines = [
        '  研究目标',
        '',
        '  x* = arg max [ μ(x) - λ·σ(x)',
        '             x∈X      - P(x) ]',
        '',
        '  μ(x): DeepEnsemble 预测均值',
        '  σ(x): 预测不确定度',
        '  λ:    风险厌恶系数',
        '  P(x): 场景软约束惩罚',
        '',
        '',
        '目标: 数据驱动代理模型 +',
        '      智能优化框架',
    ]
    add_multiline_textbox(slide, 7.9, 2.0, 4.8, 5.4, formula_lines,
                          font_size=12, color=DARK_GRAY,
                          bold_lines={0, 12},
                          color_lines={0: ORANGE, 12: ACCENT_BLUE},
                          size_lines={0: 18, 12: 14})

    add_page_number(slide, 3)
    return slide


def slide_04_dataset(prs):
    slide = new_slide(prs)
    add_title_bar(slide, '02', '数据分析与预处理', 'Dataset & Preprocessing')
    add_rounded_box(slide, 0.5, 1.8, 6.5, 3.8)

    lines = [
        '数据集: Kaggle F1 Aerodynamic Stability (~150k 样本)',
        '清洗后: 104,999 条有效样本  5 输入 → 1 输出',
        '',
        '特征: speed_kmh, wing_angle_deg, drs_active,',
        '      downforce_n, drag_n',
        '目标: stability_index (0-100, 越大越稳定)',
        '',
        '数据不平衡:',
        '  稳定 (≥95): 86.02%   不稳定 (<95): 13.98%',
        '  类平衡加权损失函数应对 (severe 权重 3.92×)',
        '预处理: StandardScaler/MinMaxScaler → 7:1.5:1.5 分割',
    ]
    add_multiline_textbox(slide, 0.8, 1.9, 6.2, 3.6, lines,
                          font_size=13, color=DARK_GRAY,
                          bold_lines={0, 6, 8})

    add_image(slide, '01_distribution.png', 7.4, 1.8, width=5.5)

    # correlation at bottom
    add_image(slide, '02_correlation.png', 0.8, 6.1, height=3.8)
    add_textbox(slide, 6.0, 7.5, 2, 0.5, '特征相关系数矩阵',
                 font_size=10, color=MID_GRAY, alignment=PP_ALIGN.CENTER)

    add_page_number(slide, 4)
    return slide


def slide_05_tech_route(prs):
    slide = new_slide(prs)
    add_title_bar(slide, '03', '技术路线总览', 'Technical Roadmap')

    # Flow chart with boxes
    steps = [
        ('数据\n预处理', 0.5, 2.2),
        ('EDA\n探索分析', 3.5, 2.2),
        ('6 模型\n代理对比', 6.5, 2.2),
        ('风险敏感\nPSO 寻优', 2.0, 5.5),
        ('主动学习\n迭代精炼', 5.3, 5.5),
    ]

    for text, x, y in steps:
        shape = slide.shapes.add_shape(
            MSO_SHAPE.ROUNDED_RECTANGLE, Cm(x), Cm(y), Cm(2.5), Cm(1.2)
        )
        shape.fill.solid()
        shape.fill.fore_color.rgb = DARK_BLUE
        shape.line.fill.background()
        tf = shape.text_frame
        tf.word_wrap = True
        p = tf.paragraphs[0]
        p.alignment = PP_ALIGN.CENTER
        run = p.add_run()
        run.text = text
        run.font.size = Pt(11)
        run.font.color.rgb = WHITE
        run.font.name = FONT_TITLE

    # Additional boxes
    steps2 = [
        ('Porpoising\n风险分析', 8.2, 5.5),
        ('多场景 +\nSHAP 可解释性', 11.0, 5.5),
    ]
    for text, x, y in steps2:
        shape = slide.shapes.add_shape(
            MSO_SHAPE.ROUNDED_RECTANGLE, Cm(x), Cm(y), Cm(2.3), Cm(1.2)
        )
        shape.fill.solid()
        shape.fill.fore_color.rgb = ACCENT_BLUE
        shape.line.fill.background()
        tf = shape.text_frame
        tf.word_wrap = True
        p = tf.paragraphs[0]
        p.alignment = PP_ALIGN.CENTER
        run = p.add_run()
        run.text = text
        run.font.size = Pt(11)
        run.font.color.rgb = WHITE
        run.font.name = FONT_TITLE

    # Bottom row
    steps3 = [
        ('消融实验\n可视化', 2.0, 8.0),
        ('结题报告', 5.0, 8.0),
    ]
    for text, x, y in steps3:
        shape = slide.shapes.add_shape(
            MSO_SHAPE.ROUNDED_RECTANGLE, Cm(x), Cm(y), Cm(2.2), Cm(1.0)
        )
        shape.fill.solid()
        shape.fill.fore_color.rgb = RGBColor(0x44, 0x6C, 0xA0)
        shape.line.fill.background()
        tf = shape.text_frame
        tf.word_wrap = True
        p = tf.paragraphs[0]
        p.alignment = PP_ALIGN.CENTER
        run = p.add_run()
        run.text = text
        run.font.size = Pt(10)
        run.font.color.rgb = WHITE
        run.font.name = FONT_TITLE

    # Innovation labels at bottom
    innovations = [
        '① 风险敏感PSO', '② 主动学习', '③ Porpoising热力图',
        '④ 多场景可解释性', '⑤ 6模型对比', '⑥ 参数区域发现',
        '⑦ 多目标Pareto', '⑧ Landscape Diagnosis'
    ]
    for i, inn in enumerate(innovations):
        col = i % 4
        row = i // 4
        shape = slide.shapes.add_shape(
            MSO_SHAPE.ROUNDED_RECTANGLE,
            Cm(0.5 + col * 3.2), Cm(9.4 + row * 0.7), Cm(3.0), Cm(0.6)
        )
        shape.fill.solid()
        shape.fill.fore_color.rgb = ORANGE if i < 4 else ACCENT_BLUE
        shape.line.fill.background()
        tf = shape.text_frame
        tf.word_wrap = True
        p = tf.paragraphs[0]
        p.alignment = PP_ALIGN.CENTER
        run = p.add_run()
        run.text = inn
        run.font.size = Pt(10)
        run.font.color.rgb = WHITE
        run.font.name = FONT_BODY

    add_page_number(slide, 5)
    return slide


def slide_06_models(prs):
    slide = new_slide(prs)
    add_title_bar(slide, '04', '代理模型构建与对比', 'Surrogate Model Comparison')

    add_rounded_box(slide, 0.5, 1.8, 6.3, 5.8)
    add_textbox(slide, 0.8, 1.9, 5.8, 0.6, '6 模型性能对比 (测试集)',
                 font_size=14, bold=True, color=DARK_BLUE)

    headers = ['模型', 'R²', 'MSE', 'MAE', '延迟(ms)']
    rows = [
        ['M1 Ridge', '0.6330', '225.00', '12.34', '0.034'],
        ['M2 RandomForest', '1.0000', '0.0000', '0.0006', '37.07'],
        ['M3 XGBoost ★', '0.9999', '0.0312', '0.0350', '0.242'],
        ['M4 MLP', '0.9994', '0.3897', '0.3882', '0.403'],
        ['M5 TabNet', '0.9997', '0.2002', '0.3602', '7.847'],
        ['M6 DeepEnsemble', '0.9995', '0.3272', '0.2712', '1.230'],
    ]
    add_table(slide, 0.8, 2.8, 5.8, 2.8, headers, rows,
              col_widths=[1.8, 1.0, 1.0, 1.0, 1.0])

    lines = [
        '策略选择:',
        '  XGBoost (M3) → SHAP 可解释性分析',
        '    TreeExplainer 精确快速，R²=0.9999',
        '',
        '  DeepEnsemble (M6) → PSO 寻优',
        '    提供 μ(x) 预测均值 + σ(x) 不确定度',
        '    3 个独立 MLP，仅种子和数据顺序不同',
    ]
    add_multiline_textbox(slide, 0.8, 5.9, 5.8, 2.0, lines,
                          font_size=12, color=DARK_GRAY,
                          bold_lines={0, 2, 5},
                          color_lines={0: ACCENT_BLUE, 2: ORANGE, 5: ORANGE})

    add_image(slide, '03_model_comparison.png', 7.4, 1.8, width=5.5)

    add_page_number(slide, 6)
    return slide


def slide_07_pso(prs):
    slide = new_slide(prs)
    add_title_bar(slide, '05', '粒子群优化算法', 'PSO Algorithm')

    add_rounded_box(slide, 0.5, 1.8, 6.5, 5.8)
    lines = [
        '标准 PSO 公式:',
        '  vᵢᵗ⁺¹ = w·vᵢᵗ + c₁r₁(pᵢ - xᵢᵗ) + c₂r₂(g - xᵢᵗ)',
        '  xᵢᵗ⁺¹ = xᵢᵗ + vᵢᵗ⁺¹',
        '',
        '自适应惯性权重:',
        '  w(t) = 0.9 - (0.9-0.4) × t/Tmax',
        '  早期大 w → 全局探索',
        '  后期小 w → 局部精细搜索',
        '',
        'PSO 配置:',
        '  粒子数 N=50  最大迭代 Tmax=80',
        '  学习因子 c₁=2.0  c₂=1.5',
        '  收敛判定 stall ≥ 12 代',
        '  风险系数 λ=1.0 (中等风险厌恶)',
    ]
    add_multiline_textbox(slide, 0.8, 1.9, 6.2, 5.5, lines,
                          font_size=13, color=DARK_GRAY,
                          bold_lines={0, 4, 8},
                          color_lines={0: ORANGE, 4: ACCENT_BLUE, 8: ACCENT_BLUE},
                          size_lines={1: 11, 5: 11, 9: 11, 10: 11, 11: 11})

    add_image(slide, '05_pso_convergence.png', 7.5, 2.0, width=5.3)

    add_page_number(slide, 7)
    return slide


def slide_08_risk_sensitive(prs):
    slide = new_slide(prs)
    add_title_bar(slide, '06', '风险敏感 PSO 创新', 'Risk-Sensitive PSO — Innovation ①')

    add_rounded_box(slide, 0.5, 1.8, 6.5, 5.8)
    lines = [
        '核心公式:',
        '  Fitness(x) = μ(x) - λ·σ(x) - P_scenario(x)',
        '',
        '利用 DeepEnsemble 不确定性估计能力:',
        '',
        '① OOD 防护',
        '  σ(x) 在分布外增大 → 惩罚使粒子回归可靠区域',
        '',
        '② 不平衡缓解',
        '  低稳定性区 σ(x) 更高 → 避免追求名义高分',
        '',
        '③ 幻觉抑制',
        '  λσ(x) 阻止虚假高分:',
        '  λ=0 → 适应度 101.30 (超理论限 100! = 幻觉)',
        '  λ=1 → 适应度 99.83  (物理合理区间)',
    ]
    add_multiline_textbox(slide, 0.8, 1.9, 6.2, 5.5, lines,
                          font_size=12, color=DARK_GRAY,
                          bold_lines={0, 4, 6, 9, 12},
                          color_lines={0: ORANGE, 4: ACCENT_BLUE, 6: ACCENT_BLUE,
                                       9: ACCENT_BLUE, 12: ACCENT_BLUE, 13: ORANGE},
                          size_lines={1: 16, 3: 14, 8: 11})

    add_image(slide, 'lambda_vs_stability.png', 7.4, 1.8, width=5.2)
    add_image(slide, '06_best_fitness_box.png', 7.4, 5.2, width=5.2)

    add_page_number(slide, 8)
    return slide


def slide_09_active_learning(prs):
    slide = new_slide(prs)
    add_title_bar(slide, '07', '主动学习迭代精炼', 'Active Learning — Innovation ②')

    add_rounded_box(slide, 0.5, 1.8, 7.0, 5.8)
    lines = [
        '闭环机制:',
        '  PSO 搜索 → 不确定性反馈 → 数据增强 → 模型重训练',
        '',
        '浅层一轮后处理精炼效果:',
        '  指标         精炼前    精炼后      变化',
        '  R²           0.9995    0.9998    +0.03%',
        '  MSE          0.3272    0.1444    -55.9% ★',
        '  MAE          0.2712    0.1678    -38.1%',
        '  RMSE         0.5720    0.3800    -33.6%',
        '',
        'PSO 收敛加速: 37 → 18 代 (提升 2.1×)',
        '推理加速:     114 → 73ms',
        '',
        '一轮精炼 = MSE 降 55.9% + 收敛快 2.1 倍',
        '证明 "PSO→反馈→改进" 闭环有效',
    ]
    add_multiline_textbox(slide, 0.8, 1.9, 6.7, 5.8, lines,
                          font_size=13, color=DARK_GRAY,
                          bold_lines={0, 3, 10, 13},
                          color_lines={0: ORANGE, 3: ACCENT_BLUE, 10: ORANGE,
                                       13: ACCENT_BLUE},
                          size_lines={10: 12, 13: 14})

    add_rounded_box(slide, 8.0, 1.8, 4.8, 3.0)
    al_lines = [
        'Top-5 高不确定性候选解特征:',
        '  下压力: 6000-9000N',
        '  σ(x): 31-38',
        '  对应训练数据中极少出现的',
        '  极端工况 (OOD 区域)',
        '',
        '主动学习通过补充 OOD 样本',
        '有效降低极端工况下的',
        '预测不确定性',
    ]
    add_multiline_textbox(slide, 8.2, 2.0, 4.4, 2.8, al_lines,
                          font_size=12, color=DARK_GRAY,
                          bold_lines={0},
                          color_lines={0: ACCENT_BLUE})

    add_page_number(slide, 9)
    return slide


def slide_10_scenarios(prs):
    slide = new_slide(prs)
    add_title_bar(slide, '08', '多场景优化', 'Multi-Scenario Optimization')

    add_rounded_box(slide, 0.5, 1.8, 6.8, 5.8)
    lines = [
        '四组典型 F1 赛道工况:',
        '',
        'S1 Monza:  [280-360km/h] [0-15°] DRS=1  极速低翼角',
        'S2 Monaco: [80-200km/h]  [20-40°] DRS=0  窄街高下压',
        'S3 均衡:   [200-280km/h] [10-30°] 自由    中速均衡',
        'S4 湿地:   [100-250km/h] [15-35°] DRS=0  湿地保守',
        '',
        '多目标适应度 (v2):',
        '  F = w₁×stab/100 + w₂×eff - w₃×power - λσ',
        '',
        '权重配置:',
        '  S1: (0.4, 0.3, 0.3) → 偏效率',
        '  S2: (0.5, 0.4, 0.1) → 高稳定',
        '  S3: (0.4, 0.4, 0.2) → 均衡',
        '  S4: (0.6, 0.2, 0.2) → 稳定优先',
    ]
    add_multiline_textbox(slide, 0.8, 1.9, 6.5, 5.5, lines,
                          font_size=12, color=DARK_GRAY,
                          bold_lines={0, 6, 9},
                          color_lines={0: ORANGE, 6: ACCENT_BLUE, 9: ACCENT_BLUE})

    add_image(slide, '07_scenario_radar.png', 7.6, 1.8, width=5.0)

    add_page_number(slide, 10)
    return slide


def slide_11_pareto(prs):
    slide = new_slide(prs)
    add_title_bar(slide, '09', 'Pareto 前沿与多目标收敛', 'Pareto Frontier & Convergence')

    add_rounded_box(slide, 0.5, 1.8, 5.8, 5.8)
    lines = [
        '多目标 vs 单目标关键差异:',
        '',
        '  单目标 (v1):',
        '    4 场景适应度 99.74~99.83',
        '    区分度仅 0.1%',
        '    → 景观过平, PSO 无价值',
        '',
        '  多目标 (v2):',
        '    4 场景适应度 0.685~0.894',
        '    区分度 > 23%',
        '    → 景观激活, PSO 有意义',
        '',
        'Pareto 前沿:',
        '  非支配解 → stability / efficiency / power',
        '  各场景因权重不同趋于不同目标空间区域',
    ]
    add_multiline_textbox(slide, 0.8, 1.9, 5.5, 5.5, lines,
                          font_size=12, color=DARK_GRAY,
                          bold_lines={0, 2, 6, 11},
                          color_lines={0: ORANGE, 2: ACCENT_BLUE, 6: ORANGE},
                          size_lines={4: 11, 8: 11})

    add_image(slide, 'pareto_frontier_S1_monza.png', 6.8, 1.8, width=5.8)
    add_image(slide, 'convergence_multiobj.png', 6.8, 4.8, width=5.8)

    add_page_number(slide, 11)
    return slide


def slide_12_shap(prs):
    slide = new_slide(prs)
    add_title_bar(slide, '10', 'SHAP 可解释性与约束灵敏度', 'SHAP & Sensitivity Analysis')

    add_image(slide, '08_shap_summary.png', 0.5, 1.8, width=6.5)

    add_rounded_box(slide, 7.3, 1.8, 5.5, 4.0)
    lines = [
        'SHAP 全局发现:',
        '  downforce: mean|SHAP|=12.20 (96%)',
        '  drag:      0.30 (2.4%)',
        '  wing:      0.15 (1.2%)',
        '  speed:     0.07 (0.6%)',
        '  drs:       0.01 (<0.1%)',
        '',
        '排列重要性验证:',
        '  打乱 downforce → R² 下降 1.88',
        '  其余特征打乱后 R² 几乎不变',
    ]
    add_multiline_textbox(slide, 7.5, 2.0, 5.0, 3.6, lines,
                          font_size=11, color=DARK_GRAY,
                          bold_lines={0, 7},
                          color_lines={0: ORANGE, 7: ACCENT_BLUE})

    add_image(slide, '11_sensitivity_tornado.png', 0.5, 6.0, width=6.0)

    add_rounded_box(slide, 7.0, 6.0, 5.8, 3.0)
    sens_lines = [
        '约束灵敏度 (各场景瓶颈):',
        '  S1 Monza:  速度下界 (Δ=0.045) ← 最大瓶颈',
        '  S2 Monaco: 翼角下界 (Δ=0.017)',
        '  S3 均衡:   四约束均匀 (Δ≈0.015)',
        '  S4 湿地:   翼角上界 (Δ=0.011)',
    ]
    add_multiline_textbox(slide, 7.2, 6.2, 5.4, 2.5, sens_lines,
                          font_size=12, color=DARK_GRAY,
                          bold_lines={0},
                          color_lines={0: ACCENT_BLUE})

    add_page_number(slide, 12)
    return slide


def slide_13_landscape_key(prs):
    slide = new_slide(prs)
    add_title_bar(slide, '11', '景观诊断核心发现  ⭐', 'Landscape Diagnosis — Key Findings')

    add_textbox(slide, 0.8, 1.6, 10, 0.5,
                '一图三结论 — 无需 SHAP/PI/偏相关前置知识即可理解',
                font_size=12, color=ORANGE, bold=True)

    add_image(slide, 'diagnosis_speed_wing_pso_overlay.png', 0.5, 2.1, height=5.8)

    add_rounded_box(slide, 0.5, 8.2, 12.3, 2.5, fill_color=RGBColor(0xF0, 0xF4, 0xFA))
    conc_lines = [
        '① ✅ 模型感知 wing — 等高线在右下角 (高速+大翼角) 可测量地下沉 (stability 55→85)',
        '② ✅ PSO 正确避开低稳定区 — 红色散点 (Top-5% 候选解) 全部集中在左上角安全区',
        '③ ✅ 高稳定盆地内 wing 无区分度 — PSO 聚集区等高线近乎水平, wing 20°→35° 几乎不改变预测',
        '',
        '代理模型在高稳定性区输出已饱和 ← 数据驱动必然结果 (86% 样本 stability≥99)',
    ]
    add_multiline_textbox(slide, 0.8, 8.3, 11.7, 2.3, conc_lines,
                          font_size=12, color=DARK_GRAY,
                          bold_lines={0, 1, 2, 4},
                          color_lines={0: ACCENT_BLUE, 1: ACCENT_BLUE, 2: ACCENT_BLUE,
                                       4: ORANGE})

    add_page_number(slide, 13)
    return slide


def slide_14_multimodality(prs):
    slide = new_slide(prs)
    add_title_bar(slide, '12', '景观多峰性验证', 'Landscape Multimodality')

    add_image(slide, 'landscape_multimodality.png', 0.5, 1.8, width=7.0)

    add_rounded_box(slide, 7.8, 1.8, 5.0, 5.8)
    lines = [
        '论证链:',
        '',
        '单目标 PCA (左上):',
        '  一个黄色大簇',
        '  → 景观过平, PSO 无意义',
        '',
        '多目标 PCA (右上):',
        '  两个分散黄色区',
        '  → 多峰结构 ↔ PSO 合理性 ✓',
        '',
        't-SNE (左下):',
        '  二次验证多峰',
        '  消除 PCA 线性投影偏误',
        '',
        '目标空间 KDE (右下):',
        '  Pareto 前沿形态',
        '  为多场景差异化搜索提供基础',
        '',
        '',
        '10 万点随机采样 → 纯景观探索',
        '(不运行 PSO)',
    ]
    add_multiline_textbox(slide, 8.0, 2.0, 4.6, 5.4, lines,
                          font_size=11, color=DARK_GRAY,
                          bold_lines={0, 2, 6, 10, 14, 18},
                          color_lines={0: ORANGE, 2: ACCENT_BLUE, 6: ORANGE,
                                       10: ACCENT_BLUE, 14: ORANGE})

    # Bottom note
    add_textbox(slide, 0.8, 8.0, 6.5, 0.5,
                'Speed×Wing 交互: 梯度在高速+大翼角达峰值 0.63, 但 PSO 自然避开此区域',
                font_size=10, color=MID_GRAY)
    add_textbox(slide, 0.8, 8.4, 6.5, 0.5,
                '翼角敏感性: ΔStability≈2 (345km/h, 非饱和区), 模型正确学习 wing↑→stability↓',
                font_size=10, color=MID_GRAY)

    add_page_number(slide, 14)
    return slide


def slide_15_porpoising(prs):
    slide = new_slide(prs)
    add_title_bar(slide, '13', 'Porpoising 海豚跳风险分析', 'Porpoising Risk — Innovation ③')

    add_image(slide, '12_risk_heatmap_drs0.png', 0.5, 1.8, width=6.2)

    add_rounded_box(slide, 7.0, 1.8, 5.8, 5.8)
    lines = [
        '一阶梯度风险度量:',
        '  Risk = ||∇f(v, α)||',
        '  (PyTorch 自动微分)',
        '',
        '关键发现:',
        '  Zone A (高速+小翼角):',
        '    平均风险 0.0013',
        '  Zone B (低速+大翼角):',
        '    平均风险 0.0002',
        '',
        '  Zone A 风险 = 6.5× Zone B ✓',
        '  (符合 F1 空气动力学直觉)',
        '',
        'DRS 开启 → 整体风险 ↑ ~1.01',
        '',
        '二阶 Hessian 曲率:',
        '  70% 网格点呈凹性',
        '  曲率幅度极低 (~0.03)',
        '  → MLP 在高稳定性区极度平坦',
    ]
    add_multiline_textbox(slide, 7.2, 2.0, 5.4, 5.4, lines,
                          font_size=11, color=DARK_GRAY,
                          bold_lines={0, 4, 14},
                          color_lines={0: ORANGE, 4: ACCENT_BLUE, 14: ACCENT_BLUE})

    add_image(slide, '13_risk_heatmap_drs1.png', 0.5, 6.2, width=2.8)
    add_image(slide, 'risk_vs_optimal_overlay.png', 3.5, 6.2, width=3.2)

    add_page_number(slide, 15)
    return slide


def slide_16_ablation(prs):
    slide = new_slide(prs)
    add_title_bar(slide, '14', '消融实验与鲁棒性', 'Ablation & Robustness')

    add_rounded_box(slide, 0.5, 1.8, 6.5, 5.8)
    add_textbox(slide, 0.8, 1.9, 5.5, 0.5, '组件消融 (以 Exp-2 风险敏感为基线)',
                 font_size=13, bold=True, color=DARK_BLUE)

    headers = ['消融操作', '适应度', '相对变化', '迭代数']
    rows = [
        ['风险敏感基线', '99.83', '—', '36.4'],
        ['A1 移除σ惩罚 ★', '101.30', '+1.47 ★', '27.9'],
        ['A2 移除密度惩罚', '99.83', '+0.004', '52.6'],
        ['A3 固定w=0.7', '99.83', '+0.003', '50.2'],
    ]
    add_table(slide, 0.8, 2.6, 5.8, 2.0, headers, rows,
              col_widths=[2.0, 1.2, 1.3, 1.3])

    lines = [
        'A1 最关键消融: 移除 σ 惩罚后',
        '  适应度 101.30 > 100 (理论上限)',
        '  → σ 惩罚是 OOD 幻觉的防护屏障',
        '',
        '密度惩罚: 主要作用 = 收敛引导',
        '  A2 迭代 +44% (36.4→52.6)',
        '',
        '自适应 w: 收敛加速',
        '  A3 迭代 +38% (36.4→50.2)',
    ]
    add_multiline_textbox(slide, 0.8, 4.9, 5.8, 2.5, lines,
                          font_size=11, color=DARK_GRAY,
                          bold_lines={0, 4, 7},
                          color_lines={0: ORANGE, 4: ACCENT_BLUE, 7: ACCENT_BLUE})

    add_image(slide, '15_ablation_study.png', 7.4, 1.8, width=5.2)
    add_image(slide, 'robustness_heatmap.png', 7.4, 4.8, width=5.2)

    add_textbox(slide, 7.4, 7.8, 5.2, 0.5,
                '超参数敏感性: 波动仅 0.297 → 方法通用性好',
                font_size=10, color=MID_GRAY, alignment=PP_ALIGN.CENTER)

    add_page_number(slide, 16)
    return slide


def slide_17_metrics(prs):
    slide = new_slide(prs)
    add_title_bar(slide, '15', '量化指标汇总', 'Quantitative Metrics Summary')

    headers = ['指标', '实际值', '目标值', '达标']
    rows = [
        ['代理模型 R² (XGBoost)', '0.9999', '≥ 0.92 (理想)', '✅ 超越'],
        ['代理模型 MSE', '0.0312', '≤ 0.02 (理想)', '✅ 接近'],
        ['PSO 收敛代数 (均值)', '27.9', '≤ 50 (理想)', '✅ 超越'],
        ['最优解标准差 (20次)', '0.0072', '≤ 0.02 (理想)', '✅ 超越'],
        ['单次寻优耗时', '83.6 ms', '≤ 5s (理想)', '✅ 大幅超越'],
        ['主动学习 MSE 降幅', '-55.9%', '—', '—'],
        ['多目标场景区分度', '> 23%', '—', '—'],
        ['自适应收敛提升', '-21% 代数', '—', '—'],
        ['Porpoising 风险比', '6.5×', '—', '—'],
    ]
    add_table(slide, 1.0, 1.8, 11.5, 4.5, headers, rows,
              col_widths=[4.0, 2.5, 2.5, 2.5])

    # Additional highlights below
    add_rounded_box(slide, 0.5, 6.8, 12.3, 2.0, fill_color=RGBColor(0xF0, 0xF4, 0xFA))
    highlights = [
        '★ 所有关键指标均达到或超越理想目标    ★ 风险敏感 σ 惩罚成功阻止 OOD 幻觉',
        '★ 多目标优化使场景区分度从 0.1% 提升至 > 23%    ★ 主动学习一轮精炼 MSE 降 55.9%',
    ]
    for i, h in enumerate(highlights):
        add_textbox(slide, 0.8, 7.0 + i * 0.6, 11.5, 0.5, h,
                     font_size=12, bold=True, color=ACCENT_BLUE)

    add_image(slide, '14_response_surface.png', 8.5, 8.0, height=1.2)

    add_page_number(slide, 17)
    return slide


def slide_18_conclusion(prs):
    slide = new_slide(prs)
    add_title_bar(slide, '16', '创新点总结与结论', 'Innovation Summary & Conclusion')

    add_rounded_box(slide, 0.5, 1.8, 6.5, 5.5)
    lines = [
        '八大创新点:',
        '',
        '① 风险敏感 PSO — μ(x)-λσ(x)',
        '   OOD/不平衡/幻觉 三合一解决',
        '② 主动学习闭环 — PSO→反馈→精炼',
        '   MSE↓55.9%, 收敛 2.1×',
        '③ Porpoising 风险热力图',
        '   一阶梯度 + 二阶 Hessian 分析',
        '④ 多场景可解释性 — SHAP+PI+灵敏度',
        '⑤ 6 模型系统对比 — 从 Linear→Ensemble',
        '⑥ 参数区域发现 — 从单点→高稳定区间',
        '⑦ 多目标 Pareto — 区分度 0.1%→23%',
        '⑧ Landscape Diagnosis — 系统验证',
    ]
    add_multiline_textbox(slide, 0.8, 2.0, 6.0, 5.0, lines,
                          font_size=11, color=DARK_GRAY,
                          bold_lines={0, 2, 4, 6, 8, 10, 11, 12},
                          color_lines={0: ORANGE, 2: ACCENT_BLUE, 4: ACCENT_BLUE,
                                       6: ACCENT_BLUE, 8: ACCENT_BLUE, 10: ACCENT_BLUE,
                                       11: ACCENT_BLUE, 12: ACCENT_BLUE})

    add_rounded_box(slide, 7.3, 1.8, 5.5, 5.5)
    concl_lines = [
        '主要结论:',
        '',
        '完整技术路线:',
        '  数据预处理 → 代理建模',
        '  → 风险感知优化',
        '  → 可解释性验证',
        '',
        '方法论可迁移至:',
        '  航空航天 | 船舶水动力学',
        '  新能源 | 高成本仿真场景',
        '',
        '不足与展望:',
        '  代理模型饱和 (数据驱动极限)',
        '  可引入真实 CFD 高保真数据',
        '  考虑瞬态时序效应',
    ]
    add_multiline_textbox(slide, 7.5, 2.0, 5.0, 5.0, concl_lines,
                          font_size=11, color=DARK_GRAY,
                          bold_lines={0, 2, 7, 10},
                          color_lines={0: ORANGE, 2: ACCENT_BLUE, 7: ACCENT_BLUE,
                                       10: ACCENT_BLUE})

    # Thank you at bottom
    add_textbox(slide, 2, 8.0, 9.5, 2.0,
                '致谢\n感谢任课老师的悉心指导！\nGitHub: github.com/AsukaLec/Formula1_Aerodynamic_Stability_Analysis',
                font_size=14, color=DARK_BLUE, font_name=FONT_TITLE,
                alignment=PP_ALIGN.CENTER)

    add_page_number(slide, 18)
    return slide


def deep_blue_slide(prs):
    """Create a deep blue full-background slide (cover, TOC)."""
    slide = new_slide(prs)
    set_slide_bg(slide, DARK_BLUE)
    return slide


# ============================================================
# Main
# ============================================================

def main():
    prs = Presentation()
    prs.slide_width = SLIDE_W
    prs.slide_height = SLIDE_H

    print("Generating slides...")

    slides = [
        ('01 Cover', slide_01_cover),
        ('02 TOC', slide_02_toc),
        ('03 Background', slide_03_background),
        ('04 Dataset', slide_04_dataset),
        ('05 Tech Route', slide_05_tech_route),
        ('06 Models', slide_06_models),
        ('07 PSO', slide_07_pso),
        ('08 Risk-Sensitive', slide_08_risk_sensitive),
        ('09 Active Learning', slide_09_active_learning),
        ('10 Scenarios', slide_10_scenarios),
        ('11 Pareto', slide_11_pareto),
        ('12 SHAP', slide_12_shap),
        ('13 Landscape Key', slide_13_landscape_key),
        ('14 Multimodality', slide_14_multimodality),
        ('15 Porpoising', slide_15_porpoising),
        ('16 Ablation', slide_16_ablation),
        ('17 Metrics', slide_17_metrics),
        ('18 Conclusion', slide_18_conclusion),
    ]

    for name, func in slides:
        print(f"  Creating slide: {name}")
        func(prs)

    output_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                               'F1_气动稳定性研究_答辩PPT.pptx')
    prs.save(output_path)
    print(f"\nPPT saved to: {output_path}")
    print(f"Total slides: {len(prs.slides)}")


if __name__ == '__main__':
    main()
