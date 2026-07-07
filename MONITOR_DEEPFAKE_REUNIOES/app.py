import json
import re
import shutil
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent


def main() -> None:
    import tkinter as tk
    from tkinter import filedialog, messagebox, ttk
    from src.ui.theme import COLORS, apply_theme, card

    root = tk.Tk()
    root.title("Verificação de reuniões e entrevistas")
    root.geometry("840x760")
    root.minsize(560, 420)
    apply_theme(root)

    header = ttk.Frame(root, style="App.TFrame")
    header.pack(fill="x", padx=38, pady=(32, 22))
    ttk.Label(header, text="Verificação de reuniões e entrevistas", style="Title.TLabel").pack(anchor="w")
    ttk.Label(
        header,
        text=(
            "Compare o participante da chamada com a referência enviada previamente "
            "e acompanhe sinais de alteração facial durante reuniões ou entrevistas."
        ),
        wraplength=750,
        style="Subtitle.TLabel",
    ).pack(anchor="w", pady=(6, 0))

    status = tk.StringVar(
        value="Selecione um candidato existente ou anexe uma nova referência."
    )
    ttk.Label(header, textvariable=status, style="Status.TLabel").pack(
        fill="x", pady=(14, 0)
    )
    active_profile_detail = tk.StringVar(value="Nenhum candidato cadastrado.")
    profile_records: list[tuple[Path, dict]] = []

    def ask_candidate_name() -> str | None:
        dialog = tk.Toplevel(root)
        dialog.title("Novo candidato")
        dialog.configure(background=COLORS["background"])
        dialog.geometry("480x250")
        dialog.resizable(False, False)
        dialog.transient(root)
        dialog.grab_set()
        result = {"name": None}

        container = card(dialog, padding=22)
        container.pack(fill="both", expand=True, padx=22, pady=22)
        ttk.Label(container, text="Identificação do candidato", style="CardTitle.TLabel").pack(anchor="w")
        ttk.Label(
            container,
            text="Informe o nome que será exibido na lista de participantes.",
            style="CardText.TLabel",
        ).pack(anchor="w", pady=(4, 12))
        entry = tk.Entry(
            container,
            font=("Segoe UI", 11),
            relief="flat",
            background=COLORS["surface_alt"],
            foreground=COLORS["text"],
            insertbackground=COLORS["text"],
        )
        entry.pack(fill="x", ipady=9)

        def confirm(_event=None):
            name = entry.get().strip()
            if name:
                result["name"] = name
                dialog.destroy()

        buttons = ttk.Frame(container, style="Card.TFrame")
        buttons.pack(fill="x", pady=(18, 0))
        ttk.Button(buttons, text="Cancelar", command=dialog.destroy, style="Secondary.TButton").pack(side="right")
        ttk.Button(buttons, text="Continuar", command=confirm, style="Primary.TButton").pack(side="right", padx=(0, 8))
        entry.bind("<Return>", confirm)
        dialog.bind("<Escape>", lambda _event: dialog.destroy())
        entry.focus_set()
        root.wait_window(dialog)
        return result["name"]

    def launch(command: list[str], message: str):
        try:
            subprocess.Popen(command, cwd=PROJECT_ROOT)
            status.set(message)
        except OSError as error:
            messagebox.showerror("Falha ao iniciar", str(error))

    def register():
        name = ask_candidate_name()
        if not name:
            return
        profile_slug = re.sub(r"[^a-zA-Z0-9_-]+", "_", name.strip()).strip("_").lower()
        existing_profile = PROJECT_ROOT / "profiles" / profile_slug / "profile.json"
        if existing_profile.exists() and not messagebox.askyesno(
            "Candidato já cadastrado",
            (
                f"Já existe um perfil para {name}.\n\n"
                "Deseja substituir a referência facial desse candidato?"
            ),
            parent=root,
        ):
            return
        messagebox.showinfo(
            "Orientações para o vídeo",
            (
                "Selecione um vídeo de aproximadamente 10 a 15 segundos contendo "
                "somente o candidato. Ele deve olhar de frente, virar o rosto para "
                "os lados e piscar, com boa iluminacao e sem filtros."
            ),
            parent=root,
        )
        video_path = filedialog.askopenfilename(
            title="Selecionar vídeo enviado pelo candidato",
            parent=root,
            filetypes=[
                ("Vídeos", "*.mp4 *.mov *.avi *.mkv *.webm"),
                ("Todos os arquivos", "*.*"),
            ],
        )
        if not video_path:
            status.set("Nenhum vídeo foi selecionado.")
            return
        selected_path = Path(video_path)
        size_mb = selected_path.stat().st_size / (1024 * 1024)
        status.set(
            f"Vídeo de {name} recebido: {selected_path.name} ({size_mb:.1f} MB)."
        )
        messagebox.showinfo(
            "Vídeo recebido",
            (
                f"Arquivo: {selected_path.name}\n"
                f"Tamanho: {size_mb:.1f} MB\n\n"
                "O arquivo foi recebido e será verificado agora."
            ),
            parent=root,
        )
        command = [
            sys.executable,
            str(PROJECT_ROOT / "register_identity.py"),
            "--name",
            name,
            "--video",
            video_path,
        ]
        try:
            process = subprocess.Popen(
                command,
                cwd=PROJECT_ROOT,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                encoding="utf-8",
                errors="replace",
            )
        except OSError as error:
            messagebox.showerror("Falha ao iniciar", str(error), parent=root)
            return

        status.set("Verificando o vídeo de referência...")

        def check_registration():
            if process.poll() is None:
                root.after(500, check_registration)
                return
            output = process.stdout.read().strip() if process.stdout is not None else ""
            if process.returncode == 0:
                status.set("Perfil do candidato criado com sucesso.")
                refresh_profiles(preferred_name=name)
                messagebox.showinfo(
                    "Cadastro concluido",
                    output or "O perfil biometrico foi criado.",
                    parent=root,
                )
            else:
                status.set("O vídeo não pôde ser usado como referência.")
                meaningful_lines = [
                    line
                    for line in output.splitlines()
                    if line.strip() and "UserWarning" not in line and "albumentations" not in line
                ]
                detail = meaningful_lines[-1] if meaningful_lines else "Falha desconhecida."
                messagebox.showerror(
                    "Vídeo não aprovado",
                    detail,
                    parent=root,
                )

        root.after(500, check_registration)

    def monitor(manual_region: bool = False):
        selected_index = candidate_selector.current() - 1
        if selected_index < 0 or selected_index >= len(profile_records):
            messagebox.showwarning(
                "Candidato necessario",
                "Cadastre ou selecione o candidato que sera monitorado.",
                parent=root,
            )
            return
        profile_path, metadata = profile_records[selected_index]
        command = [
            sys.executable,
            str(PROJECT_ROOT / "run_monitor.py"),
            "--profile",
            str(profile_path),
        ]
        if manual_region:
            command.append("--manual-region")
        launch(
            command,
            (
                f"Perfil {metadata['name']} selecionado. Recorte o quadro do participante."
                if manual_region
                else f"Perfil {metadata['name']} selecionado. Escolha a videoconferencia."
            ),
        )

    def results():
        launch(
            [sys.executable, str(PROJECT_ROOT / "evaluate_sessions.py"), "--open-dialog"],
            "Tela de conclusoes aberta.",
        )

    scroll_area = ttk.Frame(root, style="App.TFrame")
    scroll_area.pack(fill="both", expand=True, padx=(38, 20), pady=(0, 26))
    scroll_area.rowconfigure(0, weight=1)
    scroll_area.columnconfigure(0, weight=1)

    canvas = tk.Canvas(
        scroll_area,
        background=COLORS["background"],
        borderwidth=0,
        highlightthickness=0,
    )
    scrollbar = ttk.Scrollbar(
        scroll_area,
        orient="vertical",
        command=canvas.yview,
        style="Modern.Vertical.TScrollbar",
    )
    canvas.configure(yscrollcommand=scrollbar.set)
    canvas.grid(row=0, column=0, sticky="nsew")
    scrollbar.grid(row=0, column=1, sticky="ns", padx=(8, 0))

    content = ttk.Frame(canvas, style="App.TFrame")
    content_window = canvas.create_window((0, 0), window=content, anchor="nw")

    def update_scroll_region(_event=None):
        canvas.configure(scrollregion=canvas.bbox("all"))

    def update_content_width(event):
        canvas.itemconfigure(content_window, width=event.width)

    def scroll_with_mouse(event):
        direction = -1 if event.delta > 0 else 1
        canvas.yview_scroll(direction * 2, "units")

    content.bind("<Configure>", update_scroll_region)
    canvas.bind("<Configure>", update_content_width)
    canvas.bind_all("<MouseWheel>", scroll_with_mouse)

    reference_card = card(content)
    reference_card.pack(fill="x", pady=(0, 14))
    top_row = ttk.Frame(reference_card, style="Card.TFrame")
    top_row.pack(fill="x")
    ttk.Label(top_row, text="1. Referência do candidato", style="CardTitle.TLabel").pack(side="left")
    ttk.Button(
        top_row,
        text="Anexar novo vídeo",
        command=register,
        style="Primary.TButton",
    ).pack(side="right")
    ttk.Label(
        reference_card,
        text="Selecione um perfil já analisado ou adicione o vídeo recebido de um novo candidato.",
        style="CardText.TLabel",
    ).pack(anchor="w", pady=(4, 12))

    candidate_selector = ttk.Combobox(reference_card, state="readonly", style="Modern.TCombobox")
    candidate_selector.pack(fill="x")
    ttk.Label(
        reference_card,
        textvariable=active_profile_detail,
        wraplength=700,
        style="CardText.TLabel",
    ).pack(anchor="w", pady=(8, 0))

    def show_selected_profile(_event=None):
        index = candidate_selector.current() - 1
        if index < 0 or index >= len(profile_records):
            active_profile_detail.set(
                "Nenhuma referência ativa. Selecione um candidato para iniciar o acompanhamento."
            )
            return
        _, metadata = profile_records[index]
        probability = metadata.get("source_deepfake_probability")
        probability_text = (
            f" A verificação inicial apontou nível baixo ({float(probability) * 100:.2f}%)."
            if probability is not None
            else ""
        )
        method = (
            "video enviado"
            if metadata.get("registration_method") == "submitted_video"
            else "imagens"
        )
        effort_baseline = metadata.get("source_effort_probability")
        effort_text = (
            f" Baseline Effort calibrada em {float(effort_baseline) * 100:.2f}%."
            if effort_baseline is not None
            else " Baseline Effort pendente: anexe novamente o video real deste candidato."
        )
        active_profile_detail.set(
            f"Referência criada por {method}, usando {metadata.get('source_count', 0)} "
            f"quadros adequados.{probability_text}{effort_text}"
        )

    def refresh_profiles(preferred_name: str | None = None):
        nonlocal profile_records
        records = []
        for metadata_path in (PROJECT_ROOT / "profiles").glob("*/profile.json"):
            embeddings_path = metadata_path.parent / "embeddings.npy"
            if not embeddings_path.exists():
                continue
            try:
                metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                continue
            records.append((metadata_path.parent, metadata))
        profile_records = sorted(
            records,
            key=lambda item: str(item[1].get("name", "")).casefold(),
        )
        candidate_selector["values"] = ["Nenhum candidato selecionado"] + [
            str(metadata.get("name", path.name))
            for path, metadata in profile_records
        ]
        if not profile_records:
            candidate_selector.current(0)
            show_selected_profile()
            return
        selected_index = 0
        if preferred_name:
            for index, (_, metadata) in enumerate(profile_records):
                if str(metadata.get("name", "")).casefold() == preferred_name.casefold():
                    selected_index = index + 1
                    break
        candidate_selector.current(selected_index)
        show_selected_profile()

    def delete_selected_profile():
        index = candidate_selector.current() - 1
        if index < 0 or index >= len(profile_records):
            messagebox.showwarning(
                "Candidato necessário",
                "Selecione o candidato cuja referência será excluída.",
                parent=root,
            )
            return
        profile_path, metadata = profile_records[index]
        name = str(metadata.get("name", profile_path.name))
        if not messagebox.askyesno(
            "Excluir referência do candidato",
            (
                f"Deseja excluir a referência de {name}?\n\n"
                "Serão removidos o perfil facial, os dados da análise do vídeo "
                "e os metadados desse candidato. O vídeo original não está armazenado.\n\n"
                "Os históricos de sessões já realizadas serão mantidos."
            ),
            parent=root,
        ):
            return
        profiles_root = (PROJECT_ROOT / "profiles").resolve()
        resolved_profile = profile_path.resolve()
        if resolved_profile.parent != profiles_root:
            messagebox.showerror(
                "Caminho inválido",
                "A referência selecionada não pertence à pasta de perfis do projeto.",
                parent=root,
            )
            return
        try:
            shutil.rmtree(resolved_profile)
        except OSError as error:
            messagebox.showerror(
                "Não foi possível excluir",
                str(error),
                parent=root,
            )
            return
        status.set(f"A referência de {name} foi excluída.")
        refresh_profiles()
        messagebox.showinfo(
            "Referência excluída",
            f"Os dados de referência de {name} foram removidos.",
            parent=root,
        )

    candidate_selector.bind("<<ComboboxSelected>>", show_selected_profile)
    refresh_profiles()
    profile_actions = ttk.Frame(reference_card, style="Card.TFrame")
    profile_actions.pack(fill="x", pady=(12, 0))
    ttk.Button(
        profile_actions,
        text="Excluir referência selecionada",
        command=delete_selected_profile,
        style="Danger.TButton",
    ).pack(side="right")

    monitor_card = card(content)
    monitor_card.pack(fill="x", pady=(0, 14))
    ttk.Label(monitor_card, text="2. Iniciar acompanhamento", style="CardTitle.TLabel").pack(anchor="w")
    ttk.Label(
        monitor_card,
        text="Escolha como o vídeo do candidato aparece na chamada.",
        style="CardText.TLabel",
    ).pack(anchor="w", pady=(4, 12))
    actions = ttk.Frame(monitor_card, style="Card.TFrame")
    actions.pack(fill="x")
    actions.columnconfigure((0, 1), weight=1)
    ttk.Button(
        actions,
        text="Acompanhar a janela inteira",
        command=monitor,
        style="Primary.TButton",
    ).grid(row=0, column=0, sticky="ew", padx=(0, 6))
    ttk.Button(
        actions,
        text="Marcar somente o candidato",
        command=lambda: monitor(True),
        style="Secondary.TButton",
    ).grid(row=0, column=1, sticky="ew", padx=(6, 0))

    review_card = card(content)
    review_card.pack(fill="x", pady=(0, 14))
    ttk.Label(review_card, text="3. Revisar sessões", style="CardTitle.TLabel").pack(anchor="w")
    ttk.Label(
        review_card,
        text="Consulte as conclusões produzidas automaticamente para cada reunião ou entrevista.",
        style="CardText.TLabel",
    ).pack(anchor="w", pady=(4, 12))
    ttk.Button(
        review_card,
        text="Abrir histórico e conclusões",
        command=results,
        style="Secondary.TButton",
    ).pack(fill="x")

    ttk.Frame(content, style="App.TFrame", height=8).pack()
    root.mainloop()


if __name__ == "__main__":
    main()
