import json
from datetime import datetime
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY, TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (
    HRFlowable,
    Image,
    KeepTogether,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)


ROOT = Path(__file__).resolve().parent
METRICS_PATH = ROOT / "log" / "diffusion_conditional_metrics.json"
PDF_PATH = ROOT / "diffusion_report.pdf"
MARKDOWN_PATH = ROOT / "diffusion_report.md"
FONT_PATH = Path("C:/Windows/Fonts/simhei.ttf")


def register_fonts():
    pdfmetrics.registerFont(TTFont("SimHei", str(FONT_PATH)))


def build_styles():
    styles = getSampleStyleSheet()
    styles.add(
        ParagraphStyle(
            name="ChineseTitle",
            fontName="SimHei",
            fontSize=22,
            leading=30,
            alignment=TA_CENTER,
            textColor=colors.HexColor("#183B4E"),
            spaceAfter=7 * mm,
        )
    )
    styles.add(
        ParagraphStyle(
            name="ChineseSubtitle",
            fontName="SimHei",
            fontSize=10,
            leading=16,
            alignment=TA_CENTER,
            textColor=colors.HexColor("#5C6B73"),
            spaceAfter=8 * mm,
        )
    )
    styles.add(
        ParagraphStyle(
            name="ChineseHeading",
            fontName="SimHei",
            fontSize=14,
            leading=20,
            textColor=colors.HexColor("#1F5C63"),
            spaceBefore=4 * mm,
            spaceAfter=3 * mm,
            wordWrap="CJK",
        )
    )
    styles.add(
        ParagraphStyle(
            name="ChineseBody",
            fontName="SimHei",
            fontSize=9.2,
            leading=16,
            alignment=TA_JUSTIFY,
            textColor=colors.HexColor("#243238"),
            spaceAfter=2.5 * mm,
            wordWrap="CJK",
        )
    )
    styles.add(
        ParagraphStyle(
            name="ChineseSmall",
            fontName="SimHei",
            fontSize=7.6,
            leading=11,
            alignment=TA_LEFT,
            textColor=colors.HexColor("#5C6B73"),
            wordWrap="CJK",
        )
    )
    styles.add(
        ParagraphStyle(
            name="ChineseCaption",
            fontName="SimHei",
            fontSize=8,
            leading=12,
            alignment=TA_CENTER,
            textColor=colors.HexColor("#5C6B73"),
            spaceBefore=2 * mm,
            spaceAfter=3 * mm,
            wordWrap="CJK",
        )
    )
    return styles


def scaled_image(path, max_width, max_height):
    image = Image(str(path))
    ratio = min(max_width / image.imageWidth, max_height / image.imageHeight)
    image.drawWidth = image.imageWidth * ratio
    image.drawHeight = image.imageHeight * ratio
    return image


def add_page_decorations(canvas, document):
    canvas.saveState()
    page_width, _ = A4
    canvas.setStrokeColor(colors.HexColor("#C7D3D5"))
    canvas.setLineWidth(0.5)
    canvas.line(18 * mm, 15 * mm, page_width - 18 * mm, 15 * mm)
    canvas.setFont("SimHei", 7.5)
    canvas.setFillColor(colors.HexColor("#64777D"))
    canvas.drawString(
        18 * mm,
        10 * mm,
        "MLlab2026 | Bonus: Class-Conditional DDPM",
    )
    canvas.drawRightString(
        page_width - 18 * mm,
        10 * mm,
        f"Page {document.page}",
    )
    canvas.restoreState()


def section_rule():
    return HRFlowable(
        width="100%",
        thickness=0.8,
        color=colors.HexColor("#7BA6A0"),
        spaceBefore=1 * mm,
        spaceAfter=3 * mm,
    )


def build_metric_table(metrics, styles):
    data = [
        [
            Paragraph("训练设置", styles["ChineseBody"]),
            Paragraph("数值", styles["ChineseBody"]),
            Paragraph("训练设置", styles["ChineseBody"]),
            Paragraph("数值", styles["ChineseBody"]),
        ],
        [
            Paragraph("训练图像", styles["ChineseBody"]),
            str(metrics["train_images"]),
            Paragraph("验证图像", styles["ChineseBody"]),
            str(metrics["validation_images"]),
        ],
        [
            Paragraph("图像尺寸", styles["ChineseBody"]),
            f'{metrics["image_size"]} x {metrics["image_size"]}',
            Paragraph("扩散步数", styles["ChineseBody"]),
            str(metrics["num_steps"]),
        ],
        [
            Paragraph("训练轮数", styles["ChineseBody"]),
            str(metrics["epochs"]),
            Paragraph("Batch size", styles["ChineseBody"]),
            str(metrics["batch_size"]),
        ],
        [
            Paragraph("引导强度", styles["ChineseBody"]),
            f'{metrics["guidance_scale"]:.1f}',
            Paragraph("EMA decay", styles["ChineseBody"]),
            f'{metrics["ema_decay"]:.3f}',
        ],
        [
            Paragraph("最佳 epoch", styles["ChineseBody"]),
            str(metrics["best_epoch"]),
            Paragraph("最佳验证损失", styles["ChineseBody"]),
            f'{metrics["best_validation_loss"]:.5f}',
        ],
    ]
    table = Table(
        data,
        colWidths=[32 * mm, 31 * mm, 36 * mm, 36 * mm],
        hAlign="CENTER",
    )
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#DDEBE8")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.HexColor("#183B4E")),
                ("GRID", (0, 0), (-1, -1), 0.45, colors.HexColor("#B8C8C9")),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("ALIGN", (1, 1), (1, -1), "CENTER"),
                ("ALIGN", (3, 1), (3, -1), "CENTER"),
                ("TOPPADDING", (0, 0), (-1, -1), 5),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
                (
                    "ROWBACKGROUNDS",
                    (0, 1),
                    (-1, -1),
                    [colors.white, colors.HexColor("#F7FAF9")],
                ),
            ]
        )
    )
    return table


def build_pdf(metrics, styles):
    document = SimpleDocTemplate(
        str(PDF_PATH),
        pagesize=A4,
        leftMargin=18 * mm,
        rightMargin=18 * mm,
        topMargin=17 * mm,
        bottomMargin=20 * mm,
        title="MLlab2026 Diffusion Bonus Report",
        author="MLlab2026 submission",
    )
    story = []

    story.append(
        Paragraph(
            "MLlab2026 Bonus Report<br/>Class-Conditional Diffusion Model",
            styles["ChineseTitle"],
        )
    )
    story.append(
        Paragraph(
            "基于 Astilbe、Bellflower、Calendula 三类花的条件 DDPM 生成实验",
            styles["ChineseSubtitle"],
        )
    )
    story.append(section_rule())
    story.append(Paragraph("工作概述", styles["ChineseHeading"]))
    story.append(
        Paragraph(
            "本 Bonus 没有采用单纯扩大训练集的做法，而是将基础 DDPM 扩展为"
            "类别条件生成模型。模型同时接收扩散时间步和花的类别标签，"
            "因此可以从同一个噪声分布出发，指定生成 Astilbe、Bellflower "
            "或 Calendula。输入分辨率从核心 Lab 的 24 x 24 提升到 32 x 32，"
            "并使用更深的卷积残差编解码器改善空间细节。最终采用 100 步"
            "确定性 DDIM 采样替代随机 ancestral sampling，显著减少彩色噪声"
            "和棋盘状伪影。",
            styles["ChineseBody"],
        )
    )
    story.append(Paragraph("主要改进", styles["ChineseHeading"]))
    story.append(
        Paragraph(
            "1. 条件建模：时间步 embedding 与类别 embedding 相加，再通过"
            "条件 MLP 注入每个残差块，使用特征缩放和偏置调制网络。",
            styles["ChineseBody"],
        )
    )
    story.append(
        Paragraph(
            "2. Classifier-free guidance：训练时以 10% 概率随机丢弃类别标签，"
            "推理时将条件预测和无条件预测组合，引导强度设为 "
            f'{metrics["guidance_scale"]:.1f}。',
            styles["ChineseBody"],
        )
    )
    story.append(
        Paragraph(
            "3. EMA 权重：对模型参数维护指数滑动平均，验证和最终采样均使用"
            " EMA 权重，以降低单批次随机梯度带来的质量波动。",
            styles["ChineseBody"],
        )
    )
    story.append(
        Paragraph(
            "4. DDIM 采样：使用 100 步确定性反向过程。相比完整 300 步随机"
            "采样，DDIM 保留同一噪声预测网络的条件信息，同时明显抑制"
            "高频噪声残留，使三类花的颜色和轮廓更稳定。",
            styles["ChineseBody"],
        )
    )
    story.append(Paragraph("实验设置", styles["ChineseHeading"]))
    story.append(build_metric_table(metrics, styles))
    story.append(
        Paragraph(
            f"训练总耗时约 {metrics['elapsed_minutes']:.2f} 分钟；"
            f"最后一轮训练损失为 {metrics['final_train_loss']:.5f}。",
            styles["ChineseSmall"],
        )
    )
    story.append(PageBreak())

    story.append(Paragraph("训练过程", styles["ChineseHeading"]))
    story.append(
        Paragraph(
            "验证损失每隔 5 个 epoch 在独立验证集上计算。最佳模型由验证损失"
            "决定，而不是直接使用最后一轮权重，这样可降低过拟合和随机波动"
            "对最终采样的影响。",
            styles["ChineseBody"],
        )
    )
    curve_path = ROOT / "vis" / "diffusion_conditional_training_curve.png"
    curve = scaled_image(curve_path, 168 * mm, 96 * mm)
    story.append(curve)
    story.append(
        Paragraph(
            "图 1：训练损失和 EMA 验证损失曲线。",
            styles["ChineseCaption"],
        )
    )
    story.append(
        Paragraph(
            f"最佳验证损失为 {metrics['best_validation_loss']:.5f}，"
            f"出现在第 {metrics['best_epoch']} 个 epoch。"
            "损失下降说明网络学习到了从含噪图像预测噪声的方向；"
            "条件信息则决定反向采样最终落入哪一类花的区域。",
            styles["ChineseBody"],
        )
    )
    story.append(PageBreak())

    story.append(Paragraph("类别条件采样", styles["ChineseHeading"]))
    story.append(
        Paragraph(
            "下图每一行固定一个类别标签，每一列是同一类别下独立采样的结果。"
            "采样从不同的标准高斯噪声开始，并共同使用训练后的 EMA 权重与"
            " classifier-free guidance，最后通过 100 步 DDIM 得到样本。",
            styles["ChineseBody"],
        )
    )
    samples_path = ROOT / "vis" / "diffusion_conditional_samples.png"
    samples = scaled_image(samples_path, 165 * mm, 115 * mm)
    story.append(samples)
    story.append(
        Paragraph(
            "图 2：三类花的条件采样结果，从上到下依次为 Astilbe、"
            "Bellflower、Calendula。",
            styles["ChineseCaption"],
        )
    )
    story.append(
        Paragraph(
            "DDIM 结果已经能体现稳定的类别差异：第一行为粉色羽状落新妇，"
            "第二行为蓝紫色风铃草，第三行为黄色金盏花。由于图像被缩放到"
            " 32 x 32 且没有使用预训练生成模型，结果仍以整体颜色和空间结构"
            "为主，而不是高分辨率花瓣细节。",
            styles["ChineseBody"],
        )
    )
    story.append(PageBreak())

    story.append(Paragraph("反向去噪过程", styles["ChineseHeading"]))
    story.append(
        Paragraph(
            "每个类别展示一条从纯噪声到最终样本的 DDIM 反向轨迹。标题中的"
            " t 表示对应的训练扩散时间步：t 从约 300 逐步减小到 0，图像中"
            "的大尺度轮廓先出现，再逐渐形成稳定的颜色和纹理。",
            styles["ChineseBody"],
        )
    )
    process_path = ROOT / "vis" / "diffusion_conditional_denoising.png"
    process = scaled_image(process_path, 168 * mm, 104 * mm)
    story.append(process)
    story.append(
        Paragraph(
            "图 3：三类花的反向扩散轨迹，每行对应一个条件类别。",
            styles["ChineseCaption"],
        )
    )
    story.append(Paragraph("结果与局限", styles["ChineseHeading"]))
    story.append(
        Paragraph(
            "本实验完成了类别可控生成：同一个噪声生成器可以根据标签产生"
            "不同类别的样本，而不是像基础 AE/VAE 那样只能无条件采样。"
            "Classifier-free guidance 进一步提高了条件标签对最终样本的影响，"
            "EMA 改善了权重稳定性，而 100 步 DDIM 明显改善了最终视觉质量。",
            styles["ChineseBody"],
        )
    )
    story.append(
        Paragraph(
            "局限主要有三点：第一，三类花在低分辨率下差异有限；第二，"
            "训练数据规模和 GPU 显存限制了模型容量；第三，本实验用训练/验证"
            "损失和可视化评估生成过程，没有计算 FID 或人工分类准确率。"
            "因此报告结论限于课程 Bonus 的可复现实验范围。",
            styles["ChineseBody"],
        )
    )
    story.append(
        Paragraph(
            "复现实验文件：diffusion_conditional.py、"
            "log/train_diffusion_conditional.out、"
            "log/diffusion_conditional_metrics.json、"
            "model/BestModel_diffusion_conditional.pkl 和 vis/ 下的三张结果图。",
            styles["ChineseSmall"],
        )
    )
    document.build(
        story,
        onFirstPage=add_page_decorations,
        onLaterPages=add_page_decorations,
    )


def build_markdown(metrics):
    content = f"""# MLlab2026 Diffusion Bonus Report

## 任务

将基础 DDPM 扩展为三类花的类别条件扩散模型，通过类别标签控制生成结果。

## 创新点

- 时间步 embedding 与类别 embedding 联合注入残差块的缩放和偏置。
- Classifier-free guidance：训练时以 {metrics['label_dropout']:.0%} 概率丢弃标签。
- EMA 权重用于验证和最终采样。
- 使用 {metrics['sample_steps']} 步确定性 DDIM 采样减少高频伪影。
- 输入分辨率提升为 {metrics['image_size']} x {metrics['image_size']}。

## 设置

- 训练图像：{metrics['train_images']}
- 验证图像：{metrics['validation_images']}
- Epoch：{metrics['epochs']}
- 扩散步数：{metrics['num_steps']}
- 引导强度：{metrics['guidance_scale']}
- 最佳 epoch：{metrics['best_epoch']}
- 最佳验证损失：{metrics['best_validation_loss']:.5f}

## 结果

![训练曲线](vis/diffusion_conditional_training_curve.png)

![类别条件采样](vis/diffusion_conditional_samples.png)

![反向去噪过程](vis/diffusion_conditional_denoising.png)

## 局限

32 x 32 分辨率和有限数据规模使细节受限；本实验重点验证条件生成范式和
完整的训练、EMA、guidance、保存及可视化链路。
"""
    MARKDOWN_PATH.write_text(content, encoding="utf-8")


def main():
    register_fonts()
    styles = build_styles()
    metrics = json.loads(METRICS_PATH.read_text(encoding="utf-8"))
    build_pdf(metrics, styles)
    build_markdown(metrics)
    print(f"Created {PDF_PATH}")
    print(f"Created {MARKDOWN_PATH}")


if __name__ == "__main__":
    main()
