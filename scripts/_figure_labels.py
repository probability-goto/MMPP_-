"""論文の図で共通に使う軸ラベルの書き方."""


def log_scale_label(label: str) -> str:
    """縦軸を対数目盛にした指標のラベル. 元のラベルの後ろに " (log scale)" を付ける."""
    return f"{label} (log scale)"
