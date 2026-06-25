import time
import argparse

import cv2

import instagram_filters as f

FILTER_NAMES = ["original"] + list(f.FILTERS.keys()) + list(f.EFFECTS.keys())

# shared state so the mouse callback can mutate it
state = {"idx": 0, "strength": 1.5, "w": 1280, "n": len(FILTER_NAMES)}


def on_mouse(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        step = 1 if x >= state["w"] / 2 else -1
        state["idx"] = (state["idx"] + step) % state["n"]
    elif event == cv2.EVENT_RBUTTONDOWN:
        state["idx"] = (state["idx"] - 1) % state["n"]


def put_text(img, text, org, scale=0.6, color=(255, 255, 255), thick=1):
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale,
                (0, 0, 0), thick + 2, cv2.LINE_AA)
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale,
                color, thick, cv2.LINE_AA)


def render_frame(frame, idx, strength, fps=None):
    name = FILTER_NAMES[idx]
    if name == "original":
        out = frame.copy()
    else:
        out = f.apply_filter(frame, name, strength)

    h, w = out.shape[:2]

    cv2.arrowedLine(out, (40, h // 2), (12, h // 2), (255, 255, 255), 3, tipLength=0.5)
    cv2.arrowedLine(out, (w - 40, h // 2), (w - 12, h // 2), (255, 255, 255), 3, tipLength=0.5)

    put_text(out, f"[{idx}] {name.upper()}   strength: {strength:.1f}", (12, 30), 0.75)
    if fps is not None:
        put_text(out, f"{fps:4.1f} FPS", (12, 56), 0.55, (200, 255, 200))
    put_text(out, "click left/right = prev/next   +/-: strength   s: save   q: quit",
             (12, h - 14), 0.5)
    return out


def main():
    p = argparse.ArgumentParser(description="Live camera with Instagram filters")
    p.add_argument("--camera", type=int, default=0)
    p.add_argument("--width", type=int, default=1280)
    p.add_argument("--height", type=int, default=720)
    p.add_argument("--no-mirror", action="store_true")
    args = p.parse_args()

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        raise SystemExit(
            f"Cannot open camera (index {args.camera}). "
            "Try --camera 1 or close another app using the camera.")
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)

    window = "Instagram filtri - kamera"
    cv2.namedWindow(window)
    cv2.setMouseCallback(window, on_mouse)

    prev_t = time.time()
    fps = 0.0
    print("Camera opened. Click left/right on the image to change filters. q to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to read frame from camera.")
            break
        if not args.no_mirror:
            frame = cv2.flip(frame, 1)

        state["w"] = frame.shape[1]
        out = render_frame(frame, state["idx"], state["strength"], fps)
        cv2.imshow(window, out)

        now = time.time()
        dt = now - prev_t
        prev_t = now
        if dt > 0:
            fps = 0.9 * fps + 0.1 * (1.0 / dt)

        key = cv2.waitKey(1) & 0xFF
        if key in (ord('q'), 27):
            break
        elif key in (ord('n'), ord('d')):
            state["idx"] = (state["idx"] + 1) % state["n"]
        elif key in (ord('p'), ord('a')):
            state["idx"] = (state["idx"] - 1) % state["n"]
        elif ord('0') <= key <= ord('9'):
            sel = key - ord('0')
            if sel < state["n"]:
                state["idx"] = sel
        elif key in (ord('+'), ord('=')):
            state["strength"] = min(2.5, round(state["strength"] + 0.1, 1))
        elif key in (ord('-'), ord('_')):
            state["strength"] = max(0.0, round(state["strength"] - 0.1, 1))
        elif key == ord('s'):
            fn = f"snapshot_{FILTER_NAMES[state['idx']]}_{int(time.time())}.jpg"
            cv2.imwrite(fn, out)
            print("Saved:", fn)
        elif key == ord('g'):
            fn = f"grid_{int(time.time())}.jpg"
            cv2.imwrite(fn, f.make_comparison_grid(frame, strength=state["strength"]))
            print("Saved grid:", fn)

        if cv2.getWindowProperty(window, cv2.WND_PROP_VISIBLE) < 1:
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
