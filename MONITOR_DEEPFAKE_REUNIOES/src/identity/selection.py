from pathlib import Path

from src.identity.profile import IdentityProfile


def list_profiles(profiles_dir: Path) -> list[Path]:
    if not profiles_dir.exists():
        return []
    return sorted(
        path for path in profiles_dir.iterdir()
        if path.is_dir() and (path / "profile.json").exists() and (path / "embeddings.npy").exists()
    )


def choose_profile(profiles_dir: Path) -> IdentityProfile | None:
    import tkinter as tk
    from tkinter import messagebox, ttk
    from src.ui.theme import COLORS, apply_theme, card

    paths = list_profiles(profiles_dir)
    if not paths:
        raise RuntimeError(
            "Nenhum perfil cadastrado. Execute register_identity.py antes do monitor."
        )

    profiles = [IdentityProfile.load(path) for path in paths]
    selected = {"profile": None}
    root = tk.Tk()
    root.title("Selecionar identidade")
    root.geometry("600x420")
    apply_theme(root)

    header = ttk.Frame(root, style="App.TFrame")
    header.pack(fill="x", padx=28, pady=(26, 16))
    ttk.Label(header, text="Selecione o candidato", style="Title.TLabel").pack(anchor="w")
    ttk.Label(
        header,
        text="A chamada será comparada somente com a referência escolhida.",
        style="Subtitle.TLabel",
    ).pack(anchor="w", pady=(4, 0))
    frame = card(root, padding=10)
    frame.pack(fill="both", expand=True, padx=28)
    listbox = tk.Listbox(
        frame,
        font=("Segoe UI", 11),
        borderwidth=0,
        highlightthickness=0,
        activestyle="none",
        background=COLORS["surface"],
        foreground=COLORS["text"],
        selectbackground=COLORS["primary"],
        selectforeground="#FFFFFF",
    )
    listbox.pack(fill="both", expand=True)
    for profile in profiles:
        listbox.insert(tk.END, f"{profile.name}  •  referência disponível")
    listbox.selection_set(0)

    def confirm(_event=None):
        indices = listbox.curselection()
        if not indices:
            messagebox.showwarning("Seleção necessária", "Escolha um candidato.")
            return
        selected["profile"] = profiles[indices[0]]
        root.destroy()

    buttons = ttk.Frame(root, style="App.TFrame")
    buttons.pack(fill="x", padx=28, pady=20)
    ttk.Button(buttons, text="Cancelar", command=root.destroy, style="Secondary.TButton").pack(side="right")
    ttk.Button(buttons, text="Continuar", command=confirm, style="Primary.TButton").pack(side="right", padx=(0, 10))
    listbox.bind("<Double-Button-1>", confirm)
    root.bind("<Return>", confirm)
    root.mainloop()
    return selected["profile"]
