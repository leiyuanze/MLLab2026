import json
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import mm
from reportlab.platypus import (
    Image,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

from generate_diffusion_report import (
    add_page_decorations,
    build_styles,
    register_fonts,
    scaled_image,
    section_rule,
)


ROOT = Path(__file__).resolve().parent
METRICS_PATH = ROOT / "log" / "diffusion_highres_metrics.json"
PDF_PATH = ROOT / "diffusion_report.pdf"
MARKDOWN_PATH = ROOT / "diffusion_report.md"


def metric_table(metrics, styles):
    rows = [
        ["训练设置", "数值", "评估指标", "数值"],
        [
            "GPU",
            metrics["gpu"],
            "最佳 epoch",
            str(metrics["best_epoch"]),
        ],
        [
            "训练 / 验证图像",
            f'{metrics["train_images"]} / {metrics["validation_images"]}',
            "最佳验证损失",
            f'{metrics["best_validation_loss"]:.5f}',
        ],
        [
            "图像尺寸",
            f'{metrics["image_size"]} x {metrics["image_size"]}',
            "分类器验证准确率",
            f'{metrics["classifier_validation_accuracy"]:.1%}',
        ],
        [
            "扩散 / DDIM 步数",
            f'{metrics["num_steps"]} / {metrics["ddim_steps"]}',
            "类别条件一致性",
            f'{metrics["conditional_consistency"]:.1%}',
        ],
        [
            "Epoch / Batch",
            f'{metrics["completed_epochs"]} / {metrics["batch_size"]}',
            "训练时长",
            f'{metrics["elapsed_minutes"]:.1f} 分钟',
        ],
        [
            "Guidance / Dropout",
            f'{metrics["guidance_scale"]:.1f} / {metrics["label_dropout"]:.0%}',
            "采样器",
            f'DDIM {metrics["ddim_steps"]} steps',
        ],
    ]
    data = []
    for row in rows:
        data.append([Paragraph(str(cell), styles["ChineseBody"]) for cell in row])
    table = Table(
        data,
        colWidths=[38 * mm, 38 * mm, 39 * mm, 38 * mm],
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
        title="MLlab2026 Kaggle High-Resolution Diffusion Report",
        author="MLlab2026 submission",
    )
    story = []

    story.append(
        Paragraph(
            "MLlab2026 Diffusion Bonus<br/>"
            "Kaggle 64 x 64 Class-Conditional DDPM",
            styles["ChineseTitle"],
        )
    )
    story.append(
        Paragraph(
            "更高分辨率、更长训练、DDIM 采样与可量化条件一致性",
            styles["ChineseSubtitle"],
        )
    )
    story.append(section_rule())
    story.append(Paragraph("实验目标", styles["ChineseHeading"]))
    story.append(
        Paragraph(
            "本地 RTX 3050 Ti 的 4 GB 显存只能支持 32 x 32 的课程级实验。"
            f'本版本将训练迁移到 Kaggle {metrics["gpu"]}，把输出分辨率提升到'
            f' {metrics["image_size"]} x {metrics["image_size"]}，扩大条件 U-Net '
            "并训练更长时间，以检验 Diffusion 在更充足算力下的生成能力。",
            styles["ChineseBody"],
        )
    )
    story.append(Paragraph("方法设计", styles["ChineseHeading"]))
    story.append(
        Paragraph(
            "模型采用三级卷积残差 U-Net；最低层 8 x 8 特征加入空间自注意力。"
            "时间步 embedding 与类别 embedding 联合调制每个残差块，"
            "训练时使用 10% 标签 dropout 支持 classifier-free guidance。",
            styles["ChineseBody"],
        )
    )
    story.append(
        Paragraph(
            f'最终使用 EMA 权重和 {metrics["ddim_steps"]} 步确定性 DDIM。'
            "相比随机 ancestral sampling，DDIM 显著降低彩色噪声与棋盘伪影。"
            "为了不只依赖肉眼判断，本实验额外训练一个留出集分类器，"
            "比较 guidance 2.5、4.0、6.0，并按条件一致性选择最终采样设置。",
            styles["ChineseBody"],
        )
    )
    story.append(Paragraph("实验指标", styles["ChineseHeading"]))
    story.append(metric_table(metrics, styles))
    story.append(PageBreak())

    story.append(Paragraph("训练质量", styles["ChineseHeading"]))
    story.append(
        Paragraph(
            "训练损失衡量模型对随机时间步噪声的预测误差；验证损失每隔固定"
            " epoch 在 EMA 模型上计算。最佳权重由验证损失选择，避免使用"
            "训练末端的偶然波动。",
            styles["ChineseBody"],
        )
    )
    training_path = ROOT / "vis" / "diffusion_highres_training.png"
    story.append(scaled_image(training_path, 168 * mm, 96 * mm))
    story.append(
        Paragraph(
            f'图 1：Kaggle {metrics["gpu"]} 训练曲线。最佳验证损失 '
            f'{metrics["best_validation_loss"]:.5f}，出现在第 '
            f'{metrics["best_epoch"]} 个 epoch。',
            styles["ChineseCaption"],
        )
    )
    story.append(
        Paragraph(
            f'留出集分类器准确率为 '
            f'{metrics["classifier_validation_accuracy"]:.1%}，说明它具备'
            "基本的真实图像类别区分能力。对按标签生成的 DDIM 样本，"
            f'条件一致率为 {metrics["conditional_consistency"]:.1%}。'
            "该指标比只看 loss 更直接地检验类别控制是否有效；它不等同于 FID，"
            "但能发现标签条件失效。",
            styles["ChineseBody"],
        )
    )
    guidance_comparison = metrics.get("guidance_comparison", {})
    if guidance_comparison:
        story.append(
            Paragraph(
                "guidance 对比："
                + "，".join(
                    f"{scale} -> {value:.1%}"
                    for scale, value in guidance_comparison.items()
                )
                + "。最终选择 6.0。",
                styles["ChineseBody"],
            )
        )
    story.append(PageBreak())

    story.append(Paragraph("64 x 64 条件采样", styles["ChineseHeading"]))
    story.append(
        Paragraph(
            "下图每行固定一个类别，每列对应独立随机噪声。所有结果只经过"
            "训练后的 EMA 权重、classifier-free guidance 和 DDIM，不使用"
            "真实图像作为初始化或修图。",
            styles["ChineseBody"],
        )
    )
    samples_path = ROOT / "vis" / "diffusion_highres_samples.png"
    story.append(scaled_image(samples_path, 166 * mm, 118 * mm))
    story.append(
        Paragraph(
            "图 2：64 x 64 类别条件样本。",
            styles["ChineseCaption"],
        )
    )
    story.append(
        Paragraph(
            "与本地 32 x 32 版本相比，64 x 64 模型可以保留更多花瓣边缘、"
            "花心结构和背景层次。它仍然不是照片级生成，但已经能通过颜色、"
            "形状与类别语义区分三类花，达到检验 Diffusion 条件生成能力的目标。",
            styles["ChineseBody"],
        )
    )
    story.append(PageBreak())

    story.append(Paragraph("反向去噪过程", styles["ChineseHeading"]))
    story.append(
        Paragraph(
            "下图展示固定噪声逐步经过 DDIM 去噪的过程。每一行对应一个类别，"
            "从左到右扩散时间步逐渐减小，最后形成类别条件样本。",
            styles["ChineseBody"],
        )
    )
    process_path = ROOT / "vis" / "diffusion_highres_denoising.png"
    story.append(scaled_image(process_path, 168 * mm, 105 * mm))
    story.append(
        Paragraph(
            "图 3：64 x 64 DDIM 反向扩散轨迹。",
            styles["ChineseCaption"],
        )
    )
    story.append(Paragraph("结论与局限", styles["ChineseHeading"]))
    story.append(
        Paragraph(
            f'Kaggle {metrics["gpu"]} 足以把实验从 32 x 32 提升到 64 x 64，并支持更深的'
            "U-Net、更长训练和更可靠的评估。DDIM 是视觉质量改善的关键："
            "同一条件网络在随机 ancestral sampling 下存在严重高频噪声，"
            "而确定性 DDIM 能得到稳定的类别结构。",
            styles["ChineseBody"],
        )
    )
    story.append(
        Paragraph(
            "局限仍然存在：数据规模只有约两千张，64 x 64 仍低于照片级生成；"
            "条件一致性来自小型分类器，不等价于人工评估；本实验没有计算 FID。"
            "guidance 6.0 提高了类别控制，但可能加深局部过饱和纹理。"
            "因此结论限定为：在课程算力范围内，已经通过 Kaggle 证明了"
            "条件 Diffusion 的有效性，而不是训练通用高保真图像生成模型。",
            styles["ChineseBody"],
        )
    )
    story.append(
        Paragraph(
            "复现文件：kaggle/highres_diffusion/、diffusion_highres_metrics.json、"
            "train_diffusion_highres.out、BestModel_diffusion_highres.pkl "
            "以及三张高分结果图。",
            styles["ChineseSmall"],
        )
    )
    document.build(
        story,
        onFirstPage=add_page_decorations,
        onLaterPages=add_page_decorations,
    )


def build_markdown(metrics):
    content = f"""# MLlab2026 Kaggle High-Resolution Diffusion Report

## 目标

在 Kaggle {metrics['gpu']} 上训练 64 x 64 类别条件 DDPM，验证更充足算力与 DDIM 采样
是否能够明显改善生成质量。

## 方法

- 三级残差 U-Net，最低层加入自注意力。
- 时间步 embedding、类别 embedding 与 AdaGN 条件调制。
- Classifier-free guidance，标签 dropout 10%。
- guidance 对比：2.5 -> 86.1%，4.0 -> 86.1%，6.0 -> 97.2%；最终使用 6.0。
- EMA 权重与 {metrics['ddim_steps']} 步确定性 DDIM。
- 留出集分类器评估类别条件一致性。

## 指标

- GPU：{metrics['gpu']}
- 图像尺寸：{metrics['image_size']} x {metrics['image_size']}
- 验证损失：{metrics['best_validation_loss']:.5f}
- 分类器验证准确率：{metrics['classifier_validation_accuracy']:.1%}
- 条件一致性：{metrics['conditional_consistency']:.1%}

![训练曲线](vis/diffusion_highres_training.png)

![64x64 采样](vis/diffusion_highres_samples.png)

![反向去噪](vis/diffusion_highres_denoising.png)
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
