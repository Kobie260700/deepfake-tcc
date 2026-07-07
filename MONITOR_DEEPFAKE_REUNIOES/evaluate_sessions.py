import argparse
import csv
import json
import shutil
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parent


def choose_session_files() -> list[Path]:
    import tkinter as tk
    from tkinter import filedialog

    root = tk.Tk()
    root.withdraw()
    selected = filedialog.askopenfilenames(
        title="Selecionar arquivos avaliacoes.csv",
        initialdir=PROJECT_ROOT / "outputs",
        filetypes=[("Avaliacoes CSV", "avaliacoes.csv"), ("CSV", "*.csv")],
    )
    root.destroy()
    return [Path(path) for path in selected]


def load_rows(paths: list[Path]) -> list[dict]:
    rows = []
    for path in paths:
        metadata_path = path.parent / "metadata.json"
        metadata = (
            json.loads(metadata_path.read_text(encoding="utf-8"))
            if metadata_path.exists()
            else {}
        )
        with path.open(newline="", encoding="utf-8") as file:
            for row in csv.DictReader(file):
                row["sessao"] = path.parent.name
                row["rotulo"] = metadata.get("label", "")
                row["cenario"] = metadata.get("scenario", "")
                rows.append(row)
    return rows


def summarize(rows: list[dict]) -> dict:
    if not rows:
        raise ValueError("Nenhuma avaliacao encontrada.")

    risks = np.array([float(row["risco_acumulado"]) for row in rows])
    distances = np.array([float(row["distancia_identidade"]) for row in rows])
    fake_values = [
        float(row["probabilidade_deepfake"])
        for row in rows if row["probabilidade_deepfake"].strip()
    ]
    summary = {
        "avaliacoes": len(rows),
        "sessoes": len({row["sessao"] for row in rows}),
        "risco_medio": round(float(np.mean(risks)), 3),
        "risco_maximo": round(float(np.max(risks)), 3),
        "distancia_identidade_media": round(float(np.mean(distances)), 4),
        "identidade_compativel_percentual": round(
            100 * sum(row["identidade_compativel"].lower() == "true" for row in rows) / len(rows),
            2,
        ),
        "liveness_valido_percentual": round(
            100 * sum(row["liveness_valido"].lower() == "true" for row in rows) / len(rows),
            2,
        ),
        "probabilidade_deepfake_media": (
            round(float(np.mean(fake_values)), 6) if fake_values else None
        ),
        "alertas_moderados_ou_altos": sum(
            row["nivel_risco"] in {"MODERADO", "ALTO"} for row in rows
        ),
    }
    labeled = [row for row in rows if row.get("rotulo") in {"real", "fake"}]
    if labeled:
        sessions = {}
        for row in labeled:
            session = sessions.setdefault(
                row["sessao"],
                {"label": row["rotulo"], "alerts": 0, "max_risk": 0.0},
            )
            session["alerts"] += row["nivel_risco"] in {"MODERADO", "ALTO"}
            session["max_risk"] = max(session["max_risk"], float(row["risco_acumulado"]))

        session_rows = [
            {
                "rotulo": values["label"],
                "alerta": values["alerts"] > 0,
                "risco_maximo": values["max_risk"],
            }
            for values in sessions.values()
        ]
        true_positive = sum(
            row["rotulo"] == "fake" and row["alerta"] for row in session_rows
        )
        false_positive = sum(
            row["rotulo"] == "real" and row["alerta"] for row in session_rows
        )
        true_negative = sum(
            row["rotulo"] == "real" and not row["alerta"] for row in session_rows
        )
        false_negative = sum(
            row["rotulo"] == "fake" and not row["alerta"] for row in session_rows
        )
        precision = true_positive / max(true_positive + false_positive, 1)
        recall = true_positive / max(true_positive + false_negative, 1)
        summary["classificacao"] = {
            "unidade": "sessao",
            "sessoes_rotuladas": len(session_rows),
            "verdadeiros_positivos": true_positive,
            "falsos_positivos": false_positive,
            "verdadeiros_negativos": true_negative,
            "falsos_negativos": false_negative,
            "precisao": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(2 * precision * recall / max(precision + recall, 1e-9), 4),
        }
    return summary


def list_session_directories() -> list[Path]:
    output_dir = PROJECT_ROOT / "outputs"
    if not output_dir.exists():
        return []
    return sorted(
        (
            path for path in output_dir.iterdir()
            if path.is_dir() and (path / "session.json").exists()
        ),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )


def conclusion_from_assessment_file(path: Path, profile: str) -> dict:
    if not path.exists():
        return {
            "profile": profile,
            "status": "INCONCLUSIVO",
            "title": "Dados insuficientes para uma conclusão",
            "message": "A sessão terminou sem completar um período de análise.",
            "cycles": 0,
        }
    with path.open(newline="", encoding="utf-8") as file:
        rows = list(csv.DictReader(file))
    if not rows:
        return {
            "profile": profile,
            "status": "INCONCLUSIVO",
            "title": "Dados insuficientes para uma conclusão",
            "message": "A sessão terminou sem completar um período de análise.",
            "cycles": 0,
        }

    compatible = sum(row["identidade_compativel"].lower() == "true" for row in rows)
    natural_movement = sum(row["liveness_valido"].lower() == "true" for row in rows)
    fake_values = [
        float(row["probabilidade_deepfake"])
        for row in rows
        if row.get("probabilidade_deepfake", "").strip()
    ]
    calibrated_values = [
        float(row["probabilidade_calibrada"])
        for row in rows
        if row.get("probabilidade_calibrada", "").strip()
    ]
    calibrated_average = (
        sum(calibrated_values) / len(calibrated_values)
        if calibrated_values
        else None
    )
    high_manipulation = calibrated_average is not None and calibrated_average >= 0.80
    high_cycles = sum(row["nivel_risco"] == "ALTO" for row in rows)
    moderate_cycles = sum(row["nivel_risco"] == "MODERADO" for row in rows)
    maximum_risk = max(float(row["risco_acumulado"]) for row in rows)

    if high_cycles or compatible < len(rows) / 2 or high_manipulation:
        status = "ATENCAO"
        title = "Foram encontrados sinais relevantes"
        message = "A identidade observada apresentou divergência em relação à referência cadastrada."
    elif moderate_cycles:
        status = "REVISAR"
        title = "A sessão apresentou sinais que merecem atenção"
        message = "O sistema identificou ocorrências que devem ser consideradas na conclusão."
    else:
        status = "SEM_ALERTAS_RELEVANTES"
        title = "Nenhum alerta relevante foi identificado"
        message = "As análises realizadas permaneceram dentro dos parâmetros esperados."

    if high_manipulation:
        status = "ATENCAO"
        title = "Foram encontrados sinais relevantes"
        message = "O escore calibrado indicou forte possibilidade de manipulacao facial."

    return {
        "profile": profile,
        "status": status,
        "title": title,
        "message": message,
        "cycles": len(rows),
        "high_cycles": high_cycles,
        "moderate_cycles": moderate_cycles,
        "identity_compatible_percent": round(100 * compatible / len(rows), 2),
        "liveness_valid_percent": round(100 * natural_movement / len(rows), 2),
        "deepfake_average_percent": (
            round(100 * sum(fake_values) / len(fake_values), 2) if fake_values else None
        ),
        "deepfake_calibrated_average_percent": (
            round(100 * calibrated_average, 2)
            if calibrated_average is not None
            else None
        ),
        "maximum_accumulated_risk": round(maximum_risk, 2),
    }


def session_display_data(session_dir: Path) -> dict:
    session = json.loads((session_dir / "session.json").read_text(encoding="utf-8"))
    conclusion_path = session_dir / "conclusao.json"
    conclusion = None
    if conclusion_path.exists():
        conclusion = json.loads(conclusion_path.read_text(encoding="utf-8"))
    if conclusion is None or (
        conclusion.get("cycles", 0) == 0
        and (session_dir / "avaliacoes.csv").exists()
    ):
        conclusion = conclusion_from_assessment_file(
            session_dir / "avaliacoes.csv",
            session.get("profile", "Candidato"),
        )
    assessment_path = session_dir / "avaliacoes.csv"
    assessments = []
    if assessment_path.exists():
        with assessment_path.open(newline="", encoding="utf-8") as file:
            assessments = list(csv.DictReader(file))
    period_path = session_dir / "periodos.jsonl"
    periods = []
    if period_path.exists():
        periods = [
            json.loads(line)
            for line in period_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
    return {
        "session": session,
        "conclusion": conclusion,
        "assessments": assessments,
        "periods": periods,
    }


def period_summary_lines(data: dict) -> list[str]:
    assessments = iter(data.get("assessments", []))
    lines = []
    for period in data.get("periods", []):
        number = period.get("periodo", len(lines) + 1)
        if period.get("analise_completa"):
            assessment = next(assessments, None)
            if assessment is None:
                lines.append(f"Período {number}: análise concluída")
                continue
            identity = (
                "referência compatível"
                if assessment.get("identidade_compativel", "").lower() == "true"
                else "referência divergente"
            )
            movement = (
                "movimento natural"
                if assessment.get("liveness_valido", "").lower() == "true"
                else "movimento inconclusivo"
            )
            fake_value = assessment.get("probabilidade_deepfake", "").strip()
            fake_text = (
                f"alteração {float(fake_value) * 100:.2f}%"
                if fake_value
                else "alteração indisponível"
            )
            calibrated_value = assessment.get("probabilidade_calibrada", "").strip()
            delta_value = assessment.get("delta_effort", "").strip()
            if calibrated_value:
                fake_text += f"; calibrado {float(calibrated_value) * 100:.2f}%"
            if delta_value:
                fake_text += f"; delta Effort {float(delta_value) * 100:+.2f} pp"
            lines.append(
                f"Período {number}: {identity}; {movement}; {fake_text}; "
                f"atenção {float(assessment.get('risco_acumulado', 0)):.1f}"
            )
        else:
            lines.append(
                f"Período {number}: inconclusivo; {period.get('mensagem', 'dados insuficientes')}"
            )
    if not lines:
        for number, assessment in enumerate(data.get("assessments", []), start=1):
            identity = (
                "referência compatível"
                if assessment.get("identidade_compativel", "").lower() == "true"
                else "referência divergente"
            )
            lines.append(f"Período {number}: {identity}")
    return lines


def open_sessions_window() -> None:
    import tkinter as tk
    from tkinter import messagebox, ttk
    from src.ui.theme import COLORS, apply_theme, card

    sessions = list_session_directories()
    root = tk.Tk()
    root.title("Conclusões das reuniões e entrevistas")
    root.geometry("1080x700")
    root.minsize(940, 620)
    apply_theme(root)

    header = ttk.Frame(root, style="App.TFrame")
    header.pack(fill="x", padx=32, pady=(28, 20))
    ttk.Label(header, text="Histórico de sessões", style="Title.TLabel").pack(anchor="w")
    ttk.Label(
        header,
        text="Consulte a conclusão produzida automaticamente para cada reunião ou entrevista.",
        style="Subtitle.TLabel",
    ).pack(anchor="w", pady=(5, 0))

    body = ttk.Frame(root, style="App.TFrame")
    body.pack(fill="both", expand=True, padx=32, pady=(0, 28))
    body.columnconfigure(1, weight=1)
    body.rowconfigure(0, weight=1)

    history = card(body, padding=12)
    history.grid(row=0, column=0, sticky="ns", padx=(0, 14))
    ttk.Label(history, text="Sessões", style="CardTitle.TLabel").pack(anchor="w", padx=4, pady=(2, 10))
    listbox = tk.Listbox(
        history,
        width=34,
        font=("Segoe UI", 10),
        borderwidth=0,
        highlightthickness=0,
        background=COLORS["surface"],
        foreground=COLORS["text"],
        selectbackground=COLORS["primary"],
        selectforeground="#FFFFFF",
        activestyle="none",
    )
    listbox.pack(fill="both", expand=True)
    for path in sessions:
        data = session_display_data(path)
        profile = data["session"].get("profile", "Sem candidato")
        created = data["session"].get("created_at", path.name).replace("T", " ")[:16]
        listbox.insert(tk.END, f"{profile}  •  {created}")

    details = card(body, padding=20)
    details.grid(row=0, column=1, sticky="nsew")
    title_var = tk.StringVar(value="Selecione uma sessao.")
    metrics_var = tk.StringVar(value="")
    periods_var = tk.StringVar(value="")
    ttk.Label(details, text="PARECER AUTOMÁTICO", style="CardText.TLabel").pack(anchor="w")
    ttk.Label(details, textvariable=title_var, style="CardTitle.TLabel", wraplength=650).pack(anchor="w", pady=(4, 0))
    ttk.Label(details, textvariable=metrics_var, style="CardText.TLabel", wraplength=650, justify="left").pack(anchor="w", pady=(10, 18))
    ttk.Label(details, text="RESULTADOS POR PERÍODO", style="CardText.TLabel").pack(anchor="w")
    ttk.Label(
        details,
        textvariable=periods_var,
        style="CardText.TLabel",
        wraplength=650,
        justify="left",
    ).pack(anchor="w", pady=(6, 18))
    ttk.Separator(details).pack(fill="x", pady=(0, 18))
    ttk.Label(
        details,
        text=(
            "A conclusão é gerada pelo sistema a partir da correspondência facial, "
            "do movimento observado e dos sinais de alteração encontrados ao longo da chamada."
        ),
        style="CardText.TLabel",
        wraplength=650,
        justify="left",
    ).pack(anchor="w")

    selected = {"path": None}

    def show_session(_event=None):
        indices = listbox.curselection()
        if not indices:
            return
        path = sessions[indices[0]]
        selected["path"] = path
        data = session_display_data(path)
        conclusion = data["conclusion"]
        title_var.set(conclusion.get("title", "Conclusao indisponivel"))
        fake = conclusion.get("deepfake_average_percent")
        fake_text = f"{fake:.2f}%" if fake is not None else "indisponivel"
        calibrated = conclusion.get("deepfake_calibrated_average_percent")
        calibrated_text = (
            f"{calibrated:.2f}%" if calibrated is not None else "baseline pendente"
        )
        metrics_var.set(
            f"{conclusion.get('message', '')}\n\n"
            f"Períodos observados: {conclusion.get('cycles', 0)}\n"
            f"Análises completas: {conclusion.get('completed_cycles', conclusion.get('cycles', 0))}\n"
            f"Períodos inconclusivos: {conclusion.get('inconclusive_cycles', 0)}\n"
            f"Correspondência com a referência: {conclusion.get('identity_compatible_percent', 0):.2f}%\n"
            f"Movimento facial considerado natural: {conclusion.get('liveness_valid_percent', 0):.2f}%\n"
            f"Sinais médios de alteração facial: {fake_text}\n"
            f"Risco calibrado pela referência: {calibrated_text}\n"
            f"Maior nível de atenção observado: {conclusion.get('maximum_accumulated_risk', 0):.2f}"
        )
        period_lines = period_summary_lines(data)
        periods_var.set("\n".join(period_lines) if period_lines else "Nenhum período registrado.")
    def delete_session():
        indices = listbox.curselection()
        if not indices:
            messagebox.showwarning(
                "Sessão necessária",
                "Selecione a sessão que será excluída.",
                parent=root,
            )
            return
        index = indices[0]
        path = sessions[index]
        data = session_display_data(path)
        profile = data["session"].get("profile", "candidato não identificado")
        created = data["session"].get("created_at", path.name).replace("T", " ")[:19]
        if not messagebox.askyesno(
            "Excluir sessão",
            (
                f"Deseja excluir a sessão de {profile}, realizada em {created}?\n\n"
                "Serão removidos as análises, a conclusão automática e os dados técnicos. "
                "Esta ação não pode ser desfeita."
            ),
            parent=root,
        ):
            return
        outputs_root = (PROJECT_ROOT / "outputs").resolve()
        resolved_session = path.resolve()
        if resolved_session.parent != outputs_root:
            messagebox.showerror(
                "Caminho inválido",
                "A sessão selecionada não pertence ao histórico do projeto.",
                parent=root,
            )
            return
        try:
            shutil.rmtree(resolved_session)
        except OSError as error:
            messagebox.showerror(
                "Não foi possível excluir",
                str(error),
                parent=root,
            )
            return

        sessions.pop(index)
        listbox.delete(index)
        selected["path"] = None
        title_var.set("Selecione uma sessão.")
        metrics_var.set("")
        periods_var.set("")
        if sessions:
            next_index = min(index, len(sessions) - 1)
            listbox.selection_set(next_index)
            listbox.activate(next_index)
            show_session()
        messagebox.showinfo(
            "Sessão excluída",
            "O histórico e a conclusão dessa sessão foram removidos.",
            parent=root,
        )

    action_buttons = ttk.Frame(details, style="Card.TFrame")
    action_buttons.pack(fill="x", pady=(24, 0))
    ttk.Button(
        action_buttons,
        text="Excluir sessão",
        command=delete_session,
        style="Danger.TButton",
    ).pack(side="left")
    listbox.bind("<<ListboxSelect>>", show_session)
    if sessions:
        listbox.selection_set(0)
        show_session()
    root.mainloop()


def main() -> None:
    parser = argparse.ArgumentParser(description="Resume sessoes do monitor.")
    parser.add_argument("paths", nargs="*", type=Path)
    parser.add_argument("--open-dialog", action="store_true")
    args = parser.parse_args()
    if args.open_dialog:
        open_sessions_window()
        return
    paths = args.paths
    if not paths:
        raise SystemExit("Nenhum arquivo selecionado.")

    summary = summarize(load_rows(paths))
    output = PROJECT_ROOT / "results"
    output.mkdir(exist_ok=True)
    output_path = output / "resumo_sessoes.json"
    output_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"[INFO] Resumo salvo em: {output_path}")


if __name__ == "__main__":
    main()
