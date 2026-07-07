COLORS = {
    "background": "#F3F6FA",
    "surface": "#FFFFFF",
    "surface_alt": "#EAF0F7",
    "primary": "#2563EB",
    "primary_hover": "#1D4ED8",
    "text": "#172033",
    "muted": "#667085",
    "border": "#D8E0EA",
    "success": "#16835B",
    "warning": "#B66900",
    "danger": "#C53B47",
}


def apply_theme(root) -> None:
    from tkinter import ttk

    root.configure(background=COLORS["background"])
    style = ttk.Style(root)
    try:
        style.theme_use("clam")
    except Exception:
        pass

    style.configure("App.TFrame", background=COLORS["background"])
    style.configure("Card.TFrame", background=COLORS["surface"])
    style.configure(
        "Title.TLabel",
        background=COLORS["background"],
        foreground=COLORS["text"],
        font=("Segoe UI", 22, "bold"),
    )
    style.configure(
        "Subtitle.TLabel",
        background=COLORS["background"],
        foreground=COLORS["muted"],
        font=("Segoe UI", 10),
    )
    style.configure(
        "CardTitle.TLabel",
        background=COLORS["surface"],
        foreground=COLORS["text"],
        font=("Segoe UI", 11, "bold"),
    )
    style.configure(
        "CardText.TLabel",
        background=COLORS["surface"],
        foreground=COLORS["muted"],
        font=("Segoe UI", 9),
    )
    style.configure(
        "Status.TLabel",
        background=COLORS["surface_alt"],
        foreground=COLORS["text"],
        font=("Segoe UI", 9),
        padding=(12, 8),
    )
    style.configure(
        "Primary.TButton",
        background=COLORS["primary"],
        foreground="#FFFFFF",
        borderwidth=0,
        focusthickness=0,
        font=("Segoe UI", 10, "bold"),
        padding=(16, 11),
    )
    style.map(
        "Primary.TButton",
        background=[("active", COLORS["primary_hover"]), ("pressed", COLORS["primary_hover"])],
        foreground=[("disabled", "#D9E4FF")],
    )
    style.configure(
        "Secondary.TButton",
        background=COLORS["surface_alt"],
        foreground=COLORS["text"],
        borderwidth=0,
        font=("Segoe UI", 10, "bold"),
        padding=(16, 10),
    )
    style.map("Secondary.TButton", background=[("active", "#DDE7F3")])
    style.configure(
        "Danger.TButton",
        background=COLORS["danger"],
        foreground="#FFFFFF",
        borderwidth=0,
        font=("Segoe UI", 9, "bold"),
        padding=(14, 8),
    )
    style.map("Danger.TButton", background=[("active", "#A92E3A")])
    style.configure(
        "Modern.TCombobox",
        fieldbackground=COLORS["surface_alt"],
        background=COLORS["surface_alt"],
        foreground=COLORS["text"],
        bordercolor=COLORS["border"],
        padding=8,
        arrowsize=16,
    )
    style.configure("Modern.Treeview", rowheight=34, font=("Segoe UI", 9), borderwidth=0)
    style.configure(
        "Modern.Treeview.Heading",
        font=("Segoe UI", 9, "bold"),
        background=COLORS["surface_alt"],
        foreground=COLORS["text"],
    )
    style.layout(
        "Modern.Vertical.TScrollbar",
        [
            (
                "Vertical.Scrollbar.trough",
                {
                    "sticky": "ns",
                    "children": [
                        ("Vertical.Scrollbar.thumb", {"expand": "1", "sticky": "nswe"})
                    ],
                },
            )
        ],
    )
    style.configure(
        "Modern.Vertical.TScrollbar",
        background="#AAB8CB",
        troughcolor=COLORS["surface_alt"],
        bordercolor=COLORS["surface_alt"],
        lightcolor="#AAB8CB",
        darkcolor="#AAB8CB",
        relief="flat",
        borderwidth=0,
        width=9,
    )
    style.map(
        "Modern.Vertical.TScrollbar",
        background=[("active", COLORS["primary"]), ("pressed", COLORS["primary_hover"])],
        lightcolor=[("active", COLORS["primary"])],
        darkcolor=[("active", COLORS["primary"])],
    )


def card(parent, padding=18):
    from tkinter import ttk

    return ttk.Frame(parent, style="Card.TFrame", padding=padding)
