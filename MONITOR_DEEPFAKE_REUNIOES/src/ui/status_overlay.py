from __future__ import annotations

from src.capture.screen import CaptureRegion


def overlay_position(
    region: CaptureRegion,
    width: int,
    height: int,
    virtual_bounds: tuple[int, int, int, int],
) -> tuple[int, int]:
    virtual_left, virtual_top, virtual_width, virtual_height = virtual_bounds
    virtual_right = virtual_left + virtual_width
    virtual_bottom = virtual_top + virtual_height
    margin = 16

    candidates = [
        (region.left - width - margin, region.top),
        (region.left + region.width + margin, region.top),
        (region.left, region.top - height - margin),
        (region.left, region.top + region.height + margin),
    ]
    for x, y in candidates:
        if (
            virtual_left <= x
            and x + width <= virtual_right
            and virtual_top <= y
            and y + height <= virtual_bottom
        ):
            return x, y

    # Quando nao ha espaco externo, usa o canto com menor chance de cobrir o rosto.
    x = min(max(region.left + margin, virtual_left), virtual_right - width)
    y = min(max(region.top + region.height - height - margin, virtual_top), virtual_bottom - height)
    return x, y


class MonitorStatusOverlay:
    """Painel pequeno que mantem o monitor controlavel sem exibir o frame."""

    def __init__(self, profile_name: str, region: CaptureRegion):
        import tkinter as tk
        from tkinter import ttk
        from src.ui.theme import apply_theme

        self._tk = tk
        self.close_requested = False
        self.root = tk.Tk()
        self.root.title("Acompanhamento da chamada")
        self.root.attributes("-topmost", True)
        self.root.resizable(True, False)
        self.root.protocol("WM_DELETE_WINDOW", self.request_close)
        apply_theme(self.root)

        self.state = tk.StringVar(value="Preparando o acompanhamento...")
        self.result = tk.StringVar(value="Primeiro parecer em até 30 segundos.")
        self.detail = tk.StringVar(
            value="O sistema observará os movimentos naturais, sem solicitar ações."
        )

        container = ttk.Frame(self.root, style="Card.TFrame", padding=16)
        container.pack(fill="both", expand=True)
        container.columnconfigure(0, weight=1)
        ttk.Label(container, text="CHAMADA EM ACOMPANHAMENTO", style="CardText.TLabel").grid(row=0, column=0, sticky="w")
        ttk.Label(container, text=profile_name, style="CardTitle.TLabel").grid(row=1, column=0, sticky="w", pady=(2, 10))
        ttk.Separator(container).grid(row=2, column=0, sticky="ew", pady=(0, 12))
        ttk.Label(
            container,
            textvariable=self.result,
            style="CardTitle.TLabel",
            wraplength=470,
            justify="left",
        ).grid(row=3, column=0, sticky="ew")
        ttk.Label(
            container,
            textvariable=self.detail,
            style="CardText.TLabel",
            wraplength=470,
            justify="left",
        ).grid(row=4, column=0, sticky="ew", pady=(6, 12))
        ttk.Separator(container).grid(row=5, column=0, sticky="ew", pady=(0, 10))

        buttons = ttk.Frame(container, style="Card.TFrame")
        buttons.grid(row=6, column=0, sticky="ew")
        ttk.Label(buttons, textvariable=self.state, style="CardText.TLabel").pack(side="left")
        ttk.Button(
            buttons,
            text="Encerrar e gerar parecer",
            command=self.request_close,
            style="Danger.TButton",
        ).pack(side="right")

        width, height = 530, 255
        self.root.minsize(470, height)
        virtual_bounds = (
            self.root.winfo_vrootx(),
            self.root.winfo_vrooty(),
            self.root.winfo_vrootwidth(),
            self.root.winfo_vrootheight(),
        )
        x, y = overlay_position(region, width, height, virtual_bounds)
        self.root.geometry(f"{width}x{height}+{x}+{y}")
        self.pump()

    def request_close(self) -> None:
        self.close_requested = True

    def update(self, state: str, result: str, detail: str) -> None:
        self.state.set(state)
        self.result.set(result)
        self.detail.set(detail)
        self.pump()

    def pump(self) -> None:
        if self.close_requested:
            return
        try:
            self.root.update_idletasks()
            self.root.update()
        except self._tk.TclError:
            self.close_requested = True

    def close(self) -> None:
        try:
            self.root.destroy()
        except self._tk.TclError:
            pass
