from __future__ import annotations

import threading
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

from analisar import run


FILE_TYPES = [
    ("Imagens e videos", "*.jpg *.jpeg *.png *.bmp *.webp *.mp4 *.avi *.mov *.mkv *.webm *.m4v"),
    ("Todos os arquivos", "*.*"),
]


class App:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.files: list[Path] = []
        self.output = Path(__file__).resolve().parent / "resultados"
        root.title("Extracao de metricas morfologicas faciais")
        root.geometry("820x560")
        root.minsize(680, 500)
        self.status = tk.StringVar(value="Selecione ao menos uma imagem ou video seu.")
        self.output_text = tk.StringVar(value=str(self.output))
        self._build()

    def _build(self) -> None:
        frame = ttk.Frame(self.root, padding=20)
        frame.pack(fill="both", expand=True)
        ttk.Label(
            frame, text="Ficha morfologica do rosto", font=("Segoe UI", 18, "bold")
        ).pack(anchor="w")
        ttk.Label(
            frame,
            text=(
                "Extraia proporcoes faciais de uma imagem ou de varias amostras de video. "
                "O resultado pode orientar a criacao de uma identidade sintetica semelhante "
                "na estrutura, mas biometricamente distinta."
            ),
            wraplength=760,
        ).pack(anchor="w", pady=(4, 18))

        box = ttk.LabelFrame(frame, text="1. Suas imagens e videos", padding=10)
        box.pack(fill="both", expand=True, pady=6)
        self.file_list = tk.Listbox(box, height=12, selectmode="extended")
        self.file_list.pack(side="left", fill="both", expand=True)
        actions = ttk.Frame(box)
        actions.pack(side="right", fill="y", padx=(10, 0))
        ttk.Button(actions, text="Adicionar", command=self._add_files).pack(fill="x", pady=(0, 6))
        ttk.Button(actions, text="Limpar", command=self._clear_files).pack(fill="x")

        output_row = ttk.Frame(frame)
        output_row.pack(fill="x", pady=(12, 4))
        ttk.Label(output_row, text="Pasta de resultados:").pack(side="left")
        ttk.Entry(output_row, textvariable=self.output_text).pack(
            side="left", fill="x", expand=True, padx=8
        )
        ttk.Button(output_row, text="Escolher", command=self._choose_output).pack(side="right")

        self.progress = ttk.Progressbar(frame, mode="indeterminate")
        self.progress.pack(fill="x", pady=(14, 6))
        ttk.Label(frame, textvariable=self.status, wraplength=760).pack(anchor="w")
        self.run_button = ttk.Button(
            frame, text="Gerar ficha morfologica", command=self._start
        )
        self.run_button.pack(fill="x", pady=(14, 0), ipady=7)

    def _add_files(self) -> None:
        selected = [Path(path) for path in filedialog.askopenfilenames(filetypes=FILE_TYPES)]
        for path in selected:
            if path not in self.files:
                self.files.append(path)
        self._refresh()

    def _clear_files(self) -> None:
        self.files.clear()
        self._refresh()

    def _refresh(self) -> None:
        self.file_list.delete(0, tk.END)
        for path in self.files:
            self.file_list.insert(tk.END, str(path))

    def _choose_output(self) -> None:
        selected = filedialog.askdirectory(initialdir=self.output)
        if selected:
            self.output = Path(selected)
            self.output_text.set(selected)

    def _start(self) -> None:
        if not self.files:
            messagebox.showwarning("Arquivo ausente", "Selecione uma imagem ou video.")
            return
        self.output = Path(self.output_text.get()).expanduser()
        self.run_button.configure(state="disabled")
        self.progress.start(10)
        self.status.set("Carregando o modelo e extraindo as proporcoes faciais.")
        threading.Thread(target=self._worker, daemon=True).start()

    def _worker(self) -> None:
        try:
            result = run(self.files, self.output)
            message = (
                f"Ficha concluida com {result['samples']} amostra(s) valida(s).\n\n"
                f"Consulte FICHA_MORFOLOGICA.md e PROMPT_GERACAO.txt em:\n{self.output}"
            )
            self.root.after(0, self._finish, message, None)
        except Exception as error:
            self.root.after(0, self._finish, None, str(error))

    def _finish(self, message: str | None, error: str | None) -> None:
        self.progress.stop()
        self.run_button.configure(state="normal")
        if error:
            self.status.set("A extracao nao foi concluida.")
            messagebox.showerror("Erro", error)
        else:
            self.status.set("Ficha morfologica gerada com sucesso.")
            messagebox.showinfo("Concluido", message)


def main() -> None:
    root = tk.Tk()
    App(root)
    root.mainloop()


if __name__ == "__main__":
    main()
