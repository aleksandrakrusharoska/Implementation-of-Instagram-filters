import math
import cv2
import numpy as np


# ---------------------------------------------------------------------------
#  BGR(uint8) <-> RGB(float [0,1]) conversion
# ---------------------------------------------------------------------------

def _to_rgb01(bgr):
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0


def _to_bgr(rgb01):
    rgb = np.clip(rgb01 * 255.0, 0, 255).astype(np.uint8)
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)


# ---------------------------------------------------------------------------
#  CSS primitives (color-space matrix operations)
# ---------------------------------------------------------------------------

def _matrix(rgb, m):
    """Apply a 3x3 color matrix; m is 9 numbers in row-major order."""
    M = np.array(m, dtype=np.float32).reshape(3, 3)
    return np.clip(rgb @ M.T, 0, 1)


def brightness(rgb, a):
    return np.clip(rgb * a, 0, 1)


def contrast(rgb, a):
    # CSS contrast: rotates values around the midpoint 0.5
    return np.clip((rgb - 0.5) * a + 0.5, 0, 1)


def saturate(rgb, s):
    m = (0.213 + 0.787 * s, 0.715 - 0.715 * s, 0.072 - 0.072 * s,
         0.213 - 0.213 * s, 0.715 + 0.285 * s, 0.072 - 0.072 * s,
         0.213 - 0.213 * s, 0.715 - 0.715 * s, 0.072 + 0.928 * s)
    return _matrix(rgb, m)


def hue_rotate(rgb, deg):
    c = math.cos(math.radians(deg))
    s = math.sin(math.radians(deg))
    m = (0.213 + c * 0.787 - s * 0.213, 0.715 - c * 0.715 - s * 0.715, 0.072 - c * 0.072 + s * 0.928,
         0.213 - c * 0.213 + s * 0.143, 0.715 + c * 0.285 + s * 0.140, 0.072 - c * 0.072 - s * 0.283,
         0.213 - c * 0.213 - s * 0.787, 0.715 - c * 0.715 + s * 0.715, 0.072 + c * 0.928 + s * 0.072)
    return _matrix(rgb, m)


def sepia(rgb, a):
    m1 = 1 - min(a, 1)
    m = (0.393 + 0.607 * m1, 0.769 - 0.769 * m1, 0.189 - 0.189 * m1,
         0.349 - 0.349 * m1, 0.686 + 0.314 * m1, 0.168 - 0.168 * m1,
         0.272 - 0.272 * m1, 0.534 - 0.534 * m1, 0.131 + 0.869 * m1)
    return _matrix(rgb, m)


def grayscale(rgb, a):
    g = 1 - min(a, 1)
    m = (0.2126 + 0.7874 * g, 0.7152 - 0.7152 * g, 0.0722 - 0.0722 * g,
         0.2126 - 0.2126 * g, 0.7152 + 0.2848 * g, 0.0722 - 0.0722 * g,
         0.2126 - 0.2126 * g, 0.7152 - 0.7152 * g, 0.0722 + 0.9278 * g)
    return _matrix(rgb, m)


# ---------------------------------------------------------------------------
#  Blend modes (cb = backdrop, cs = source layer; values in [0,1])
# ---------------------------------------------------------------------------

def _multiply(cb, cs):
    return cb * cs


def _screen(cb, cs):
    return cb + cs - cb * cs


def _hard_light(cb, cs):
    return np.where(cs <= 0.5, 2 * cb * cs, 1 - 2 * (1 - cb) * (1 - cs))


def _overlay(cb, cs):
    # overlay(cb, cs) = hard_light(cs, cb)
    return np.where(cb <= 0.5, 2 * cb * cs, 1 - 2 * (1 - cb) * (1 - cs))


def _soft_light(cb, cs):
    cbc = np.clip(cb, 0, 1)
    D = np.where(cbc <= 0.25, ((16 * cbc - 12) * cbc + 4) * cbc, np.sqrt(cbc))
    return np.where(cs <= 0.5,
                    cb - (1 - 2 * cs) * cb * (1 - cb),
                    cb + (2 * cs - 1) * (D - cb))


def _color_dodge(cb, cs):
    denom = np.clip(1 - cs, 1e-6, 1)
    out = np.minimum(1.0, cb / denom)
    out = np.where(cb <= 0, 0.0, out)
    out = np.where(cs >= 1, 1.0, out)
    return out


def _darken(cb, cs):
    return np.minimum(cb, cs)


def _exclusion(cb, cs):
    return cb + cs - 2 * cb * cs


def _solid(color255):
    return np.array(color255, dtype=np.float32) / 255.0


def blend_solid(cb, color255, mode, opacity=1.0):
    """Blend backdrop cb with a solid color layer using the given mode and opacity."""
    cs = _solid(color255)
    b = np.clip(mode(cb, cs), 0, 1)
    return np.clip(cb * (1 - opacity) + b * opacity, 0, 1)


def _radial_mask(h, w, length=0.0, scale=1.0, center=(0.5, 0.5)):
    """Radial gradient mask: 1 at center, fading to 0 at edges."""
    cx, cy = center
    rw_left, rw_right = w * cx, w * (1 - cx)
    rh_top, rh_bottom = h * cy, h * (1 - cy)
    x = np.linspace(-rw_left, rw_right, w)
    y = np.linspace(-rh_top, rh_bottom, h)[:, None]
    r = math.sqrt(max(rw_left, rw_right) ** 2 + max(rh_top, rh_bottom) ** 2)
    base = max(scale - length, 0.001)
    m = np.sqrt(x ** 2 + y ** 2) / r
    m = (m - length) / base
    m = 1 - m
    return np.clip(m, 0, 1)[..., None]


# ---------------------------------------------------------------------------
#  Filters (internal: RGB[0,1] -> RGB[0,1])
# ---------------------------------------------------------------------------

def _clarendon(rgb):
    cr = blend_solid(rgb, (127, 187, 227), _overlay, 0.2)
    cr = contrast(cr, 1.2)
    cr = saturate(cr, 1.35)
    return cr


def _gingham(rgb):
    cr = np.clip(_soft_light(rgb, _solid((230, 230, 250))), 0, 1)
    cr = brightness(cr, 1.05)
    cr = hue_rotate(cr, -10)
    return cr


def _juno(rgb):
    cr = sepia(rgb, 0.35)
    cr = contrast(cr, 1.15)
    cr = brightness(cr, 1.15)
    cr = saturate(cr, 1.8)
    return cr


def _lark(rgb):
    cm1 = np.clip(_color_dodge(rgb, _solid((34, 37, 63))), 0, 1)
    cs2 = _solid((242, 242, 242))
    cr = np.clip(cm1 * 0.2 + _darken(cm1, cs2) * 0.8, 0, 1)
    cr = contrast(cr, 0.9)
    return cr


def _mayfair(rgb):
    h, w = rgb.shape[:2]
    pos = (0.4, 0.4)
    cm1 = blend_solid(rgb, (255, 255, 255), _overlay, 0.8)
    cm2 = blend_solid(rgb, (255, 200, 200), _overlay, 0.6)
    cm3 = np.clip(_overlay(rgb, _solid((17, 17, 17))), 0, 1)
    m1 = _radial_mask(h, w, scale=0.3, center=pos)
    cs = cm1 * m1 + cm2 * (1 - m1)
    m2 = _radial_mask(h, w, length=0.3, scale=0.6, center=pos)
    cs = cs * m2 + cm3 * (1 - m2)
    cr = np.clip(rgb * 0.6 + cs * 0.4, 0, 1)
    cr = contrast(cr, 1.1)
    cr = saturate(cr, 1.1)
    return cr


def _sierra(rgb):
    cr = contrast(rgb, 0.8)
    cr = saturate(cr, 1.2)
    cr = sepia(cr, 0.15)
    return cr


def _valencia(rgb):
    cs = np.clip(_exclusion(rgb, _solid((58, 3, 57))), 0, 1)
    cr = np.clip(rgb * 0.5 + cs * 0.5, 0, 1)
    cr = contrast(cr, 1.08)
    cr = brightness(cr, 1.08)
    cr = sepia(cr, 0.08)
    return cr


def _walden(rgb):
    cs = np.clip(_screen(rgb, _solid((0, 68, 204))), 0, 1)
    cr = np.clip(rgb * 0.7 + cs * 0.3, 0, 1)
    cr = brightness(cr, 1.1)
    cr = hue_rotate(cr, -10)
    cr = sepia(cr, 0.3)
    cr = saturate(cr, 1.6)
    return cr


def _rainbow(rgb):
    h, w = rgb.shape[:2]
    xx, yy = np.meshgrid(np.linspace(0, 1, w), np.linspace(0, 1, h))
    t = xx * 0.75 + (1 - yy) * 0.25
    hue = (0.82 * (1 - t)) % 1.0
    H = (hue * 179).astype(np.uint8)
    S = np.full_like(H, 230)
    V = np.full_like(H, 255)
    layer = cv2.cvtColor(cv2.merge([H, S, V]), cv2.COLOR_HSV2RGB).astype(np.float32) / 255.0
    op = 0.60
    return np.clip(rgb * (1 - op) + _overlay(rgb, layer) * op, 0, 1)


def _wrap(fn):
    def f(img):
        return _to_bgr(fn(_to_rgb01(img)))

    return f


FILTERS = {
    "clarendon": _wrap(_clarendon), "gingham": _wrap(_gingham),
    "juno": _wrap(_juno), "lark": _wrap(_lark),
    "mayfair": _wrap(_mayfair), "sierra": _wrap(_sierra),
    "valencia": _wrap(_valencia), "walden": _wrap(_walden),
    "rainbow": _wrap(_rainbow),
}


def apply_filter(img, name, strength=1.0):
    """
    Apply a filter with optional strength blending:
        result = original + strength * (filtered - original)
    strength=1.0 is the canonical look; >1 is stronger, <1 is softer.
    For effects, strength is ignored.
    """
    if name in EFFECTS:
        return EFFECTS[name](img)
    out = FILTERS[name](img)
    if abs(strength - 1.0) > 1e-6:
        out = cv2.addWeighted(out, float(strength), img, 1.0 - float(strength), 0)
    return out


EFFECTS = {}


# ---------------------------------------------------------------------------
#  Histogram and comparison grid
# ---------------------------------------------------------------------------

def histogram_image(img, size=(256, 200)):
    w, h = size
    canvas = np.full((h, w, 3), 255, np.uint8)
    colors = [(255, 0, 0), (0, 180, 0), (0, 0, 255)]
    for ch, col in enumerate(colors):
        hist = cv2.calcHist([img], [ch], None, [256], [0, 256]).flatten()
        cv2.normalize(hist, hist, 0, h - 1, cv2.NORM_MINMAX)
        pts = [(x * w // 256, h - 1 - int(hist[x])) for x in range(256)]
        for i in range(1, 256):
            cv2.line(canvas, pts[i - 1], pts[i], col, 1, cv2.LINE_AA)
    return canvas


def make_comparison_grid(img, strength=1.0, cols=3, cell=320, pad=8, bg=245):
    label_h = 26

    def fit(im):
        scale = min(cell / im.shape[1], cell / im.shape[0])
        rs = cv2.resize(im, (int(im.shape[1] * scale), int(im.shape[0] * scale)))
        canvas = np.full((cell, cell, 3), bg, np.uint8)
        y0 = (cell - rs.shape[0]) // 2
        x0 = (cell - rs.shape[1]) // 2
        canvas[y0:y0 + rs.shape[0], x0:x0 + rs.shape[1]] = rs
        return canvas

    items = [("Original", img)] + [(n.capitalize(), apply_filter(img, n, strength))
                                   for n in FILTERS]
    rows = (len(items) + cols - 1) // cols
    gw = cols * cell + (cols + 1) * pad
    gh = rows * (cell + label_h) + (rows + 1) * pad
    grid = np.full((gh, gw, 3), bg, np.uint8)
    for i, (name, im) in enumerate(items):
        r, c = divmod(i, cols)
        x = pad + c * (cell + pad)
        y = pad + r * (cell + label_h + pad)
        cv2.putText(grid, name, (x + 4, y + 18), cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (30, 30, 30), 1, cv2.LINE_AA)
        grid[y + label_h:y + label_h + cell, x:x + cell] = fit(im)
    return grid


def main():
    import argparse
    import os
    p = argparse.ArgumentParser(description="Instagram filters (Python + OpenCV)")
    p.add_argument("input")
    p.add_argument("-f", "--filter", default="all",
                   choices=list(FILTERS.keys()) + list(EFFECTS.keys()) + ["all"])
    p.add_argument("-o", "--output", default="rezultati")
    p.add_argument("-s", "--strength", type=float, default=1.0)
    p.add_argument("--grid", action="store_true")
    p.add_argument("--hist", metavar="PATH")
    args = p.parse_args()

    img = cv2.imread(args.input)
    if img is None:
        raise SystemExit(f"Cannot read image: {args.input}")

    if args.grid:
        out = args.output if args.output.lower().endswith((".jpg", ".png")) else "sporedba.jpg"
        cv2.imwrite(out, make_comparison_grid(img, strength=args.strength))
        print("Saved:", out)
        return

    if args.filter == "all":
        os.makedirs(args.output, exist_ok=True)
        for name in list(FILTERS) + list(EFFECTS):
            path = os.path.join(args.output, f"{name}.jpg")
            cv2.imwrite(path, apply_filter(img, name, args.strength))
            print("Saved:", path)
    else:
        result = apply_filter(img, args.filter, args.strength)
        cv2.imwrite(args.output, result)
        print("Saved:", args.output)
        if args.hist:
            cv2.imwrite(args.hist, histogram_image(result))
            print("Saved histogram:", args.hist)


if __name__ == "__main__":
    main()
