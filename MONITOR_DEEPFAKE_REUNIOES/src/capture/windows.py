import ctypes
from ctypes import wintypes
from dataclasses import dataclass

from src.capture.screen import CaptureRegion


user32 = ctypes.windll.user32

# Sem assinaturas explicitas, ctypes pode truncar HWNDs em processos Windows
# de 64 bits. Isso fazia algumas janelas, especialmente Chrome/Meet, sumirem.
user32.EnumWindows.argtypes = [
    ctypes.WINFUNCTYPE(wintypes.BOOL, wintypes.HWND, wintypes.LPARAM),
    wintypes.LPARAM,
]
user32.EnumWindows.restype = wintypes.BOOL
user32.IsWindowVisible.argtypes = [wintypes.HWND]
user32.IsWindowVisible.restype = wintypes.BOOL
user32.IsIconic.argtypes = [wintypes.HWND]
user32.IsIconic.restype = wintypes.BOOL
user32.GetWindowTextLengthW.argtypes = [wintypes.HWND]
user32.GetWindowTextLengthW.restype = ctypes.c_int
user32.GetWindowTextW.argtypes = [wintypes.HWND, wintypes.LPWSTR, ctypes.c_int]
user32.GetWindowTextW.restype = ctypes.c_int
user32.GetWindowRect.argtypes = [wintypes.HWND, ctypes.POINTER(wintypes.RECT)]
user32.GetWindowRect.restype = wintypes.BOOL
user32.ShowWindow.argtypes = [wintypes.HWND, ctypes.c_int]
user32.ShowWindow.restype = wintypes.BOOL
user32.SetForegroundWindow.argtypes = [wintypes.HWND]
user32.SetForegroundWindow.restype = wintypes.BOOL


@dataclass(frozen=True)
class DesktopWindow:
    handle: int
    title: str
    region: CaptureRegion


def window_region(handle: int) -> CaptureRegion | None:
    rect = wintypes.RECT()
    if not user32.GetWindowRect(handle, ctypes.byref(rect)):
        return None

    width = rect.right - rect.left
    height = rect.bottom - rect.top
    if width <= 100 or height <= 100:
        return None
    return CaptureRegion(rect.left, rect.top, width, height)


def list_visible_windows() -> list[DesktopWindow]:
    windows = []

    enum_callback_type = user32.EnumWindows.argtypes[0]

    @enum_callback_type
    def callback(handle, _):
        if not user32.IsWindowVisible(handle) or user32.IsIconic(handle):
            return True

        title_length = user32.GetWindowTextLengthW(handle)
        if title_length == 0:
            return True

        buffer = ctypes.create_unicode_buffer(title_length + 1)
        user32.GetWindowTextW(handle, buffer, title_length + 1)
        title = buffer.value.strip()
        region = window_region(handle)
        if title and region is not None:
            windows.append(DesktopWindow(int(handle), title, region))
        return True

    user32.EnumWindows(callback, 0)
    return sorted(windows, key=lambda window: window.title.casefold())


def activate_window(handle: int) -> None:
    user32.ShowWindow(handle, 9)
    user32.SetForegroundWindow(handle)


def choose_window() -> DesktopWindow | None:
    import tkinter as tk
    from tkinter import messagebox, ttk
    from src.ui.theme import COLORS, apply_theme, card

    windows = list_visible_windows()
    if not windows:
        raise RuntimeError("Nenhuma janela visivel foi encontrada.")

    selected = {"window": None}
    root = tk.Tk()
    root.title("Selecionar videoconferencia")
    root.geometry("820x560")
    root.minsize(700, 480)
    apply_theme(root)

    header = ttk.Frame(root, style="App.TFrame")
    header.pack(fill="x", padx=30, pady=(28, 18))
    ttk.Label(header, text="Onde está a chamada?", style="Title.TLabel").pack(anchor="w")
    ttk.Label(
        header,
        text=(
            "Escolha a janela do Meet, Zoom, Teams ou Discord. No navegador, deixe a aba "
            "da chamada aberta antes de continuar."
        ),
        wraplength=750,
        style="Subtitle.TLabel",
    ).pack(anchor="w", pady=(5, 0))

    frame = card(root, padding=12)
    frame.pack(fill="both", expand=True, padx=30)
    scrollbar = ttk.Scrollbar(frame)
    scrollbar.pack(side="right", fill="y")
    listbox = tk.Listbox(
        frame,
        font=("Segoe UI", 10),
        activestyle="none",
        borderwidth=0,
        highlightthickness=0,
        background=COLORS["surface"],
        foreground=COLORS["text"],
        selectbackground=COLORS["primary"],
        selectforeground="#FFFFFF",
        yscrollcommand=scrollbar.set,
    )
    listbox.pack(side="left", fill="both", expand=True)
    scrollbar.config(command=listbox.yview)

    current_windows = []

    def populate():
        nonlocal current_windows
        current_windows = list_visible_windows()
        listbox.delete(0, tk.END)
        for window in current_windows:
            listbox.insert(
                tk.END,
                f"{window.title}  [{window.region.width}x{window.region.height}]",
            )
        for index, window in enumerate(current_windows):
            title = window.title.casefold()
            if any(name in title for name in ("meet", "zoom", "teams", "discord")):
                listbox.selection_set(index)
                listbox.see(index)
                break

    def confirm(_event=None):
        indices = listbox.curselection()
        if not indices:
            messagebox.showwarning("Seleção necessária", "Escolha uma janela da lista.")
            return
        selected["window"] = current_windows[indices[0]]
        root.destroy()

    buttons = ttk.Frame(root, style="App.TFrame")
    buttons.pack(fill="x", padx=30, pady=22)
    ttk.Button(buttons, text="Atualizar janelas", command=populate, style="Secondary.TButton").pack(side="left")
    ttk.Button(buttons, text="Cancelar", command=root.destroy, style="Secondary.TButton").pack(side="right")
    ttk.Button(buttons, text="Usar esta janela", command=confirm, style="Primary.TButton").pack(side="right", padx=(0, 10))

    listbox.bind("<Double-Button-1>", confirm)
    root.bind("<Return>", confirm)
    root.bind("<Escape>", lambda _event: root.destroy())
    populate()
    root.mainloop()
    return selected["window"]
