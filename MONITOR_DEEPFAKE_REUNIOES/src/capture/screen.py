from dataclasses import dataclass
from typing import Callable

import cv2
import numpy as np
from PIL import Image, ImageGrab


@dataclass(frozen=True)
class CaptureRegion:
    left: int
    top: int
    width: int
    height: int

    @property
    def bbox(self) -> tuple[int, int, int, int]:
        return (
            self.left,
            self.top,
            self.left + self.width,
            self.top + self.height,
        )


def _virtual_screen_bounds() -> tuple[int, int, int, int]:
    import tkinter as tk

    root = tk.Tk()
    root.withdraw()
    root.update_idletasks()
    left = root.winfo_vrootx()
    top = root.winfo_vrooty()
    width = root.winfo_vrootwidth()
    height = root.winfo_vrootheight()
    root.destroy()
    return left, top, width, height


def _pil_to_bgr(image) -> np.ndarray:
    rgb = np.asarray(image.convert("RGB"))
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)


def select_capture_region(
    max_preview_width: int = 1600,
    max_preview_height: int = 900,
    base_region: CaptureRegion | None = None,
) -> CaptureRegion:
    import tkinter as tk
    from PIL import ImageTk

    if base_region is None:
        left, top, width, height = _virtual_screen_bounds()
    else:
        left = base_region.left
        top = base_region.top
        width = base_region.width
        height = base_region.height
    screenshot = _pil_to_bgr(
        ImageGrab.grab(bbox=(left, top, left + width, top + height), all_screens=True)
    )

    scale = min(max_preview_width / width, max_preview_height / height, 1.0)
    preview = cv2.resize(screenshot, None, fx=scale, fy=scale) if scale < 1.0 else screenshot
    preview_rgb = cv2.cvtColor(preview, cv2.COLOR_BGR2RGB)

    root = tk.Tk()
    root.title("Selecionar participante")
    root.attributes("-topmost", True)
    root.resizable(False, False)
    photo = ImageTk.PhotoImage(image=Image.fromarray(preview_rgb))
    canvas = tk.Canvas(root, width=preview.shape[1], height=preview.shape[0], cursor="cross")
    canvas.pack()
    canvas.create_image(0, 0, image=photo, anchor="nw")
    canvas.create_rectangle(0, 0, preview.shape[1], 48, fill="#111111", outline="")
    canvas.create_text(
        18,
        24,
        anchor="w",
        fill="white",
        font=("Segoe UI", 12, "bold"),
        text="Clique e arraste sobre o video do candidato. ENTER confirma; ESC cancela.",
    )
    selection = {"start": None, "rect": None, "coords": None}

    def start_selection(event):
        selection["start"] = (event.x, event.y)
        if selection["rect"] is not None:
            canvas.delete(selection["rect"])
        selection["rect"] = canvas.create_rectangle(
            event.x, event.y, event.x, event.y, outline="#00ff66", width=3
        )

    def drag_selection(event):
        if selection["start"] is not None:
            x1, y1 = selection["start"]
            canvas.coords(selection["rect"], x1, y1, event.x, event.y)

    def finish_selection(event):
        if selection["start"] is None:
            return
        x1, y1 = selection["start"]
        selection["coords"] = (
            min(x1, event.x), min(y1, event.y),
            max(x1, event.x), max(y1, event.y),
        )

    def confirm(_event=None):
        if selection["coords"] is not None:
            root.destroy()

    def cancel(_event=None):
        selection["coords"] = None
        root.destroy()

    canvas.bind("<ButtonPress-1>", start_selection)
    canvas.bind("<B1-Motion>", drag_selection)
    canvas.bind("<ButtonRelease-1>", finish_selection)
    root.bind("<Return>", confirm)
    root.bind("<Escape>", cancel)
    root.protocol("WM_DELETE_WINDOW", cancel)
    root.mainloop()

    if selection["coords"] is None:
        raise RuntimeError("Selecao de regiao cancelada.")

    x1, y1, x2, y2 = selection["coords"]
    selected_width = x2 - x1
    selected_height = y2 - y1
    if selected_width < 80 or selected_height < 80:
        raise RuntimeError("A regiao selecionada e pequena demais.")

    return CaptureRegion(
        left=left + round(x1 / scale),
        top=top + round(y1 / scale),
        width=round(selected_width / scale),
        height=round(selected_height / scale),
    )


class ScreenCapture:
    def __init__(
        self,
        region: CaptureRegion,
        region_provider: Callable[[], CaptureRegion | None] | None = None,
    ):
        self.region = region
        self.region_provider = region_provider

    def read(self) -> np.ndarray:
        if self.region_provider is not None:
            updated_region = self.region_provider()
            if updated_region is not None:
                self.region = updated_region
        try:
            image = ImageGrab.grab(bbox=self.region.bbox, all_screens=True)
        except OSError as error:
            raise RuntimeError(
                "A captura de tela falhou. Execute o monitor em uma sessao "
                "interativa do Windows e confirme que a area selecionada esta visivel."
            ) from error
        return _pil_to_bgr(image)
