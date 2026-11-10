"""FruitScope 网页。运行：python -m streamlit run web/app.py"""

from __future__ import annotations

from html import escape
from io import BytesIO
import logging
import os
from pathlib import Path
import sys

from PIL import Image, ImageOps, UnidentifiedImageError
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import config
from web.knowledge import FRUIT_NOTES, PRODUCE_REFERENCE


MAX_BYTES = 10 * 1024 * 1024
MAX_PIXELS = 24_000_000
MAX_SIDE = 6_000
EXAMPLES = {
    "苹果": "01_correct_apple.jpg",
    "香蕉": "02_correct_banana.jpg",
    "葡萄": "03_correct_grapes.jpg",
    "橙子": "06_correct_orange.jpg",
}
LOGGER = logging.getLogger(__name__)

HERO = """
<div class="fs-masthead">
  <div class="fs-wordmark"><span class="fs-emblem" aria-hidden="true"></span>
    <span><strong>FruitScope</strong><small>水果观察站</small></span>
  </div>
  <a href="#fs-guide">使用指南 <span aria-hidden="true">↘</span></a>
</div>
<header class="fs-hero" aria-labelledby="fs-title">
  <p class="fs-eyebrow"><span></span> 图像识别 <i></i> 自然观察</p>
  <h1 id="fs-title">这是什么水果？</h1>
  <p>上传一张照片，看看模型觉得它最像哪一种。<br>再找找果皮、果形和果梗，验证你的判断。</p>
</header>
"""

VIEWFINDER = """
<div class="fs-viewfinder">
  <div class="fs-reticle" aria-hidden="true"><i></i><b></b></div>
  <strong>照片预览</strong>
  <p>把整个水果拍进来，特征会更清楚。</p>
</div>
"""

PHOTO_TIPS = """
<section class="fs-photo-tips" aria-label="拍摄建议">
  <div class="fs-tip-heading"><span>拍照小提示</span><strong>让水果更好认</strong></div>
  <div><b>01</b><strong>一次拍一种</strong><p>让最想识别的水果占据画面中心。</p></div>
  <div><b>02</b><strong>保留整个果形</strong><p>稍微退后一点，把果皮和果梗也拍进去。</p></div>
  <div><b>03</b><strong>找一处亮光</strong><p>靠近窗边拍，避开背光和重滤镜。</p></div>
</section>
"""


def checkpoint_path() -> Path:
    if configured := os.environ.get("FRUITSCOPE_CHECKPOINT"):
        return ROOT / Path(configured).expanduser()
    export = config.MODEL_DIR / "fruitscope.pth"
    if export.is_file():
        return export
    return config.DEMO_DEFAULT_CKPT


@st.cache_resource(max_entries=2, show_spinner=False)
def load_predictor(path: Path, revision: tuple[int, int]):
    """缓存模型。文件被替换时，修改时间与大小会更新缓存键。"""
    from src.predict import Predictor

    return Predictor(path, topk=3, device="cpu")


def read_preview(raw: bytes) -> Image.Image:
    if not raw or len(raw) > MAX_BYTES:
        raise ValueError("这张照片太大了。请选一张小于或等于 10 MB 的照片。")
    try:
        with Image.open(BytesIO(raw)) as image:
            if image.format not in {"JPEG", "PNG"}:
                raise ValueError("请换一张 JPG 或 PNG 照片。")
            if image.width * image.height > MAX_PIXELS or max(image.size) > MAX_SIDE:
                raise ValueError("照片尺寸太大。缩小后再上传，最长边最多 6000 像素，总像素不超过 2400 万。")
            preview = ImageOps.exif_transpose(image).convert("RGB")
            preview.thumbnail((1000, 1000))
            return preview
    except (UnidentifiedImageError, OSError, Image.DecompressionBombError) as exc:
        raise ValueError("这张照片打不开。试试重新保存为 JPG 或 PNG，再上传。") from exc


def format_score(probability: float) -> str:
    if probability >= 0.999:
        return "≥99.9%"
    if 0 < probability < 0.001:
        return "<0.1%"
    return f"{probability:.1%}"


def show_result(result: dict) -> None:
    best = result["topk"][0]
    if result["rejected"]:
        st.warning("这次模型没把握。试试换个角度，让一种水果完整地出现在画面里。")
    elif best["label"] == config.OTHER_CLASS:
        st.info("模型更倾向于「其他」。如果你拍的是水果，可以从下方的使用指南看看它是否在识别范围内。")
    else:
        score = escape(format_score(best["prob"]))
        fill = min(100, max(0, best["prob"] * 100))
        st.html(
            '<div class="fs-match"><span>这张照片最像</span>'
            f'<h2>{escape(best["name_cn"])}</h2>'
            f'<span>模型得分</span><strong>{score}</strong>'
            f'<div class="fs-meter" aria-hidden="true"><i style="--fs-score: {fill:.4f}%"></i></div></div>'
        )

    with st.expander("查看前三个候选", expanded=result["rejected"] or best["label"] == config.OTHER_CLASS):
        for candidate in result["topk"]:
            label = "其他" if candidate["label"] == config.OTHER_CLASS else candidate["name_cn"]
            st.progress(candidate["prob"], text=f'{label} · {format_score(candidate["prob"])}')
        st.caption("分数表示模型更倾向于哪一项。外观相似时，高分也可能认错。")


def choose_example() -> bytes | None:
    st.caption("手边没有水果？点一个水果名，试试示例照片。")
    with st.container(key="example_gallery"):
        for column, (name, filename) in zip(st.columns(len(EXAMPLES), gap="small"), EXAMPLES.items()):
            path = config.DEMO_SAMPLES_DIR / filename
            if path.is_file():
                with column:
                    st.image(str(path), width="stretch")
                    selected = st.session_state.get("example") == name
                    st.button(
                        f"✓ {name}" if selected else name,
                        key=f"sample_{filename}",
                        width="stretch",
                        on_click=st.session_state.update,
                        args=({"example": name},),
                    )
    if filename := EXAMPLES.get(st.session_state.get("example")):
        try:
            return (config.DEMO_SAMPLES_DIR / filename).read_bytes()
        except OSError:
            st.error("这张示例图暂时打不开。选另一张，或用你自己的照片试试。")
    return None


def show_input(checkpoint: Path, revision: tuple[int, int] | None) -> None:
    with st.container(key="scan_panel", border=True):
        left, right = st.columns((1, 1), gap="medium", vertical_alignment="center")
        raw = None
        with left:
            st.html('<p class="fs-kicker"><b>01</b> 选择照片</p>')
            st.subheader("开始一次水果观察")
            mode = st.radio(
                "照片来源", ("上传照片", "试试示例"), horizontal=True, key="image_mode",
                label_visibility="collapsed",
            )
            if mode == "上传照片":
                upload = st.file_uploader(
                    "选择或拖入水果照片", type=["jpg", "jpeg", "png"], label_visibility="collapsed"
                )
                if upload is not None:
                    raw = upload.getvalue()
            else:
                raw = choose_example()

            preview = None
            if raw is not None:
                try:
                    preview = read_preview(raw)
                except ValueError as exc:
                    st.error(str(exc))

            if st.button("开始识别", type="primary", disabled=preview is None or revision is None, width="stretch"):
                with st.spinner("正在比较照片里的特征…"):
                    try:
                        result = load_predictor(checkpoint, revision).predict(BytesIO(raw))
                    except Exception:
                        LOGGER.exception("水果识别失败")
                        st.error("这次识别没完成。可以再试一次，或换一张照片。")
                    else:
                        st.session_state.observation = (checkpoint, revision, preview, result)
                        st.rerun()

        with right, st.container(key="input_preview"):
            if preview is None:
                st.html(VIEWFINDER)
            else:
                st.image(preview, width="stretch", caption="你选的照片")


def reset_observation() -> None:
    st.session_state.pop("observation", None)
    st.session_state.pop("example", None)


def show_observation(preview: Image.Image, result: dict) -> None:
    with st.container(key="observation_panel", border=True):
        photo, summary = st.columns((1.05, 1), gap="large", vertical_alignment="center")
        with photo, st.container(key="observation_photo"):
            st.image(preview, width="stretch", caption="你观察的水果")
        with summary:
            show_result(result)

    with st.container(key="again"):
        st.button("↺  识别另一张", on_click=reset_observation, type="secondary")

    best = result["topk"][0]
    if not result["rejected"] and (note := FRUIT_NOTES.get(best["label"])):
        with st.container(key="knowledge_card", border=True):
            st.html('<p class="fs-kicker"><b>02</b> 对照特征</p>')
            st.subheader(f'观察笔记 · {best["name_cn"]}')
            facts, tips = st.columns(2)
            facts.markdown(f"**在照片里找一找**  \n{note[0]}")
            tips.markdown(f"**买回家怎么挑、怎么放**  \n{note[1]}")
            st.caption(f"[科普资料来源 · UC Davis]({PRODUCE_REFERENCE})")


def main() -> None:
    st.set_page_config(page_title="FruitScope · 水果观察站", layout="wide")
    st.html(Path(__file__).with_name("style.css"))
    st.html(HERO)

    checkpoint = checkpoint_path()
    try:
        stat = checkpoint.stat() if checkpoint.is_file() else None
    except OSError:
        LOGGER.exception("无法读取模型文件状态")
        stat = None
    revision = (stat.st_mtime_ns, stat.st_size) if stat else None
    if revision is None:
        st.info("识别暂时不可用。你仍然可以选择照片并预览。")

    observation = st.session_state.get("observation")
    if observation and observation[:2] == (checkpoint, revision):
        show_observation(*observation[2:])
    else:
        st.session_state.pop("observation", None)
        show_input(checkpoint, revision)
        st.html(PHOTO_TIPS)
    st.html('<div id="fs-guide"></div>')
    with st.expander("使用指南 · 哪些水果能识别？"):
        st.markdown("**目前能识别的 10 种水果**")
        st.write("、".join(config.class_cn(label) for label in config.FRUIT_CLASSES))
        st.markdown(
            "模型只学过上面这些水果类别。遇到不熟悉的照片，它可能给出「其他」，"
            "也可能把相似的水果认错，比如把柠檬认成橙子。看到结果后，你也可以自己对照外观。"
        )
        st.caption(
            "模型：项目训练的 MobileNetV3。训练数据与示例图来源："
            "[Fruit and Vegetable Image Recognition](https://www.kaggle.com/datasets/kritikseth/fruit-and-vegetable-image-recognition)。"
        )
    st.html('<footer class="fs-footer"><strong>FruitScope</strong><span>水果观察站 · 课堂与日常观察</span></footer>')


if __name__ == "__main__":
    main()
