import argparse
import ctypes
import time
from pathlib import Path

import cv2
import numpy as np

from src.analysis_window import AnalysisWindow
from src.capture.screen import CaptureRegion, ScreenCapture, select_capture_region
from src.capture.windows import activate_window, choose_window, window_region
from src.config import MonitorConfig
from src.deepfake_detection.base import calibrate_result
from src.deepfake_detection.factory import load_detector
from src.face_tracking.analyzer import FaceMonitorAnalyzer, draw_face_analysis
from src.identity.profile import IdentityProfile, IdentityVerifier
from src.identity.selection import choose_profile
from src.liveness.passive import analyze_liveness, normalized_face_crop, normalized_landmarks
from src.risk.engine import RiskEngine
from src.session_conclusion import build_final_conclusion, friendly_cycle_result
from src.storage.session import SessionRecorder
from src.ui.status_overlay import MonitorStatusOverlay


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config" / "monitor.json"
OUTPUT_ROOT = PROJECT_ROOT / "outputs"
WINDOW_NAME = "Monitor de Identidade e Deepfake"
PROFILES_DIR = PROJECT_ROOT / "profiles"
MODELS_DIR = PROJECT_ROOT / "models"


def draw_text(frame, text: str, line: int, color=(255, 255, 255), scale=0.62) -> None:
    cv2.putText(frame, text, (16, line), cv2.FONT_HERSHEY_SIMPLEX, scale, color, 2)


def key_pressed(virtual_key: int) -> bool:
    return bool(ctypes.windll.user32.GetAsyncKeyState(virtual_key) & 0x0001)


def choose_capture_target(
    use_manual_selection: bool,
) -> tuple[CaptureRegion, int | None, CaptureRegion | None]:
    window = choose_window()
    if window is None:
        raise RuntimeError("Selecao de janela cancelada.")
    activate_window(window.handle)
    print(f"[INFO] Janela selecionada: {window.title}")
    if use_manual_selection:
        # A janela escolhida e ativada pelo programa antes da captura; o usuario
        # nao precisa correr para trocar do VS Code para a videoconferencia.
        time.sleep(0.25)
        selected_region = select_capture_region(base_region=window.region)
        return selected_region, window.handle, window.region
    return window.region, window.handle, None


def tracked_region_provider(
    window_handle: int,
    selected_region: CaptureRegion,
    original_window_region: CaptureRegion | None,
):
    if original_window_region is None:
        return lambda: window_region(window_handle)

    relative_left = (
        selected_region.left - original_window_region.left
    ) / original_window_region.width
    relative_top = (
        selected_region.top - original_window_region.top
    ) / original_window_region.height
    relative_width = selected_region.width / original_window_region.width
    relative_height = selected_region.height / original_window_region.height

    def provider() -> CaptureRegion | None:
        current_window_region = window_region(window_handle)
        if current_window_region is None:
            return None
        return CaptureRegion(
            current_window_region.left + round(current_window_region.width * relative_left),
            current_window_region.top + round(current_window_region.height * relative_top),
            max(1, round(current_window_region.width * relative_width)),
            max(1, round(current_window_region.height * relative_height)),
        )

    return provider


def run(
    config_path: Path,
    fixed_region: CaptureRegion | None = None,
    use_manual_selection: bool = False,
    profile_path: Path | None = None,
) -> None:
    config = MonitorConfig.load(config_path)
    profile = (
        IdentityProfile.load(profile_path)
        if profile_path is not None
        else choose_profile(PROFILES_DIR)
    )
    if profile is None:
        raise RuntimeError("Selecao de identidade cancelada.")
    if fixed_region is not None:
        region, window_handle, original_window_region = fixed_region, None, None
    else:
        region, window_handle, original_window_region = choose_capture_target(
            use_manual_selection
        )
    capture = ScreenCapture(
        region,
        region_provider=(
            tracked_region_provider(window_handle, region, original_window_region)
            if window_handle is not None
            else None
        ),
    )
    analyzer = FaceMonitorAnalyzer(config)
    identity_verifier = IdentityVerifier(profile, config.identity_distance_threshold)
    deepfake_detector = load_detector(MODELS_DIR)
    risk_engine = RiskEngine(
        config.risk_alert_threshold,
        config.risk_high_threshold,
        config.identity_distance_threshold,
        config.identity_warning_ratio,
    )
    analysis_window = AnalysisWindow(config.analysis_window_samples)
    recorder = SessionRecorder(OUTPUT_ROOT, config.store_embeddings)
    detector_description = deepfake_detector.predict([])
    recorder.write_session_info({
        "profile": profile.name,
        "profile_created_at": profile.created_at,
        "profile_registration_method": profile.registration_method,
        "profile_source_deepfake_probability": profile.source_deepfake_probability,
        "profile_source_effort_probability": profile.source_effort_probability,
        "detector": detector_description.model_name,
        "detector_available": detector_description.available,
        "capture_region": {
            "left": region.left,
            "top": region.top,
            "width": region.width,
            "height": region.height,
        },
        "configuration": config.__dict__,
    })

    print(f"[INFO] Regiao: {region}")
    print(f"[INFO] Identidade esperada: {profile.name}")
    print(f"[INFO] Detector: {detector_description.model_name}")
    print(f"[INFO] Sessao: {recorder.session_dir}")
    print("[INFO] O recrutador encerra pelo painel ou pela tecla F9.")

    status_overlay = (
        MonitorStatusOverlay(profile.name, region)
        if config.show_status_overlay and not config.show_preview
        else None
    )
    if status_overlay is not None and window_handle is not None:
        activate_window(window_handle)

    frame_period = 1.0 / max(config.target_capture_fps, 1.0)
    last_frame_time = time.perf_counter()
    last_sample_time = time.monotonic()
    current_fps = 0.0
    periods_completed = 0
    analyses_completed = 0
    latest_identity = None
    latest_deepfake = None
    latest_liveness = None
    latest_risk = None
    session_assessments = []
    latest_period_message = None

    try:
        while True:
            loop_start = time.perf_counter()
            frame = capture.read()
            face = analyzer.select_identity_face(frame, identity_verifier.reference)
            now = time.monotonic()
            sample_status = "aguardando rosto adequado"

            if face is not None:
                quality = analyzer.evaluate_quality(
                    frame,
                    face,
                    minimum_sharpness=config.monitor_minimum_sharpness,
                )
                draw_face_analysis(frame, face, quality)
                embedding = analyzer.embedding(face)
                gray_crop = normalized_face_crop(frame, face)
                color_crop = analyzer.face_crop(frame, face, margin_ratio=0.30)
                aligned_color_crop = analyzer.aligned_face_crop(frame, face)
                analysis_window.add(
                    embedding,
                    normalized_landmarks(face),
                    gray_crop,
                    color_crop,
                    aligned_color_crop,
                    quality.adequate,
                )
                if not quality.adequate:
                    analysis_window.add_quality_reasons(quality.reasons)
                if quality.adequate:
                    sample_status = "coletando janela de analise"
                else:
                    sample_status = ", ".join(quality.reasons)
            else:
                analysis_window.add_invalid("rosto_nao_detectado")
                draw_text(frame, "ROSTO NAO DETECTADO", 36, (0, 0, 255), 0.75)

            if now - last_sample_time >= config.sample_interval_seconds:
                periods_completed += 1
                if analysis_window.ready():
                    snapshot = analysis_window.snapshot()
                    latest_identity = identity_verifier.verify(snapshot.embeddings)
                    latest_liveness = analyze_liveness(
                        snapshot.landmarks,
                        snapshot.gray_crops,
                        config.minimum_temporal_variation,
                        config.minimum_landmark_movement,
                        config.maximum_landmark_movement,
                    )
                    latest_deepfake = deepfake_detector.predict(
                        snapshot.color_crops,
                        snapshot.aligned_color_crops,
                    )
                    latest_deepfake = calibrate_result(
                        latest_deepfake,
                        profile.source_effort_probability,
                        config.effort_delta_warning,
                        config.effort_delta_alert,
                    )
                    latest_risk = risk_engine.assess(
                        latest_identity,
                        latest_deepfake,
                        latest_liveness,
                        snapshot.quality_ratio,
                        None,
                    )
                    representative_quality = quality if face is not None else None
                    if representative_quality is not None:
                        recorder.record(
                            frame,
                            representative_quality,
                            np.median(np.stack(snapshot.embeddings), axis=0),
                        )
                    recorder.record_assessment(
                        latest_identity, latest_deepfake, latest_liveness, latest_risk
                    )
                    session_assessments.append({
                        "identity": latest_identity,
                        "deepfake": latest_deepfake,
                        "liveness": latest_liveness,
                        "risk": latest_risk,
                    })
                    latest_period_message = friendly_cycle_result(latest_risk)
                    print(
                        f"[AVALIACAO {analyses_completed + 1}] "
                        f"identidade={latest_identity.distance:.2f} "
                        f"deepfake={latest_deepfake.probability} "
                        f"liveness={latest_liveness.valid} "
                        f"risco={latest_risk.level}/{latest_risk.accumulated:.1f}"
                    )
                    analyses_completed += 1
                    sample_status = "avaliacao atualizada"
                    recorder.record_period(
                        periods_completed,
                        True,
                        analysis_window.sample_count,
                        analysis_window.total_frames,
                        latest_period_message[0],
                    )
                else:
                    valid_samples = analysis_window.sample_count
                    main_reason = analysis_window.main_rejection_reason()
                    reason_messages = {
                        "rosto_nao_detectado": "O rosto não permaneceu visível na área monitorada.",
                        "rosto_pequeno": "O rosto apareceu pequeno demais para uma análise confiável.",
                        "baixa_confianca": "O enquadramento dificultou a identificação do rosto.",
                        "iluminacao_inadequada": "A iluminação do rosto ficou fora do nível adequado.",
                        "baixa_nitidez": "A imagem do rosto ficou desfocada ou comprimida.",
                        "qualidade_insuficiente": "A imagem facial apresentou qualidade instável.",
                    }
                    reason = reason_messages.get(
                        main_reason,
                        "O rosto apareceu por pouco tempo ou com qualidade instável.",
                    )
                    latest_period_message = (
                        "Análise parcial neste período",
                        reason + " O acompanhamento continuará normalmente.",
                    )
                    sample_status = "periodo inconclusivo"
                    recorder.record_period(
                        periods_completed,
                        False,
                        valid_samples,
                        analysis_window.total_frames,
                        reason,
                    )
                    print(f"[PERIODO {periods_completed}] inconclusivo: {reason}")
                analysis_window.reset()
                # Inference can take a few seconds, especially while Effort is
                # loading. The next 30-second period starts after it finishes.
                last_sample_time = time.monotonic()

            elapsed = loop_start - last_frame_time
            if elapsed > 0:
                instant_fps = 1.0 / elapsed
                current_fps = (
                    instant_fps
                    if current_fps == 0
                    else 0.85 * current_fps + 0.15 * instant_fps
                )
            last_frame_time = loop_start

            draw_text(frame, f"Perfil: {profile.name}", 68, (255, 255, 0))
            draw_text(frame, f"FPS: {current_fps:.1f} | Periodos: {periods_completed}", 98, (255, 255, 0))
            draw_text(frame, f"Estado: {sample_status}", 128, (255, 255, 0))
            if latest_identity is not None:
                identity_color = (0, 220, 0) if latest_identity.compatible else (0, 0, 255)
                draw_text(
                    frame,
                    f"Identidade: {'COMPATIVEL' if latest_identity.compatible else 'DIVERGENTE'} "
                    f"(D={latest_identity.distance:.2f})",
                    158,
                    identity_color,
                )
            if latest_deepfake is not None:
                if latest_deepfake.effort_delta is not None:
                    detector_text = (
                        f"Effort: {latest_deepfake.effort_probability * 100:.1f}% | "
                        f"delta: {latest_deepfake.effort_delta * 100:+.1f} pp | "
                        f"calibrado: {(latest_deepfake.calibrated_probability or 0) * 100:.1f}%"
                    )
                elif latest_deepfake.effort_probability is not None:
                    detector_text = (
                        f"Effort: {latest_deepfake.effort_probability * 100:.1f}% "
                        "(baseline pendente)"
                    )
                else:
                    detector_text = (
                        f"Manipulacao: {latest_deepfake.probability * 100:.1f}%"
                        if latest_deepfake.probability is not None
                        else "Manipulacao: detector nao configurado"
                    )
                draw_text(frame, detector_text, 188, (180, 220, 255))
            if latest_liveness is not None:
                draw_text(
                    frame,
                    f"Liveness passivo: {'VALIDO' if latest_liveness.valid else 'INCONCLUSIVO'}",
                    218,
                    (0, 220, 0) if latest_liveness.valid else (0, 165, 255),
                )
            if latest_risk is not None:
                risk_color = {
                    "BAIXO": (0, 220, 0),
                    "MODERADO": (0, 165, 255),
                    "ALTO": (0, 0, 255),
                }[latest_risk.level]
                draw_text(
                    frame,
                    f"RISCO: {latest_risk.level} ({latest_risk.accumulated:.1f})",
                    252,
                    risk_color,
                    0.8,
                )
            draw_text(
                frame,
                "Q/ESC/F9: encerrar e gerar conclusao",
                frame.shape[0] - 18,
                (200, 200, 200),
                0.5,
            )

            seconds_to_next = max(
                0,
                config.sample_interval_seconds - (now - last_sample_time),
            )
            if latest_period_message is None:
                result_text = f"Primeira conclusao em aproximadamente {seconds_to_next:.0f}s."
                detail_text = "Observando identidade, alterações faciais e movimentos naturais."
            else:
                result_text, detail_text = latest_period_message
            if status_overlay is not None:
                status_overlay.update(
                    f"Períodos: {periods_completed}  |  Próxima atualização: {seconds_to_next:.0f}s",
                    result_text,
                    detail_text,
                )

            if config.show_preview:
                cv2.imshow(WINDOW_NAME, frame)
                key = cv2.waitKey(1) & 0xFF
            else:
                key = 255
                if status_overlay is not None and status_overlay.close_requested:
                    key = ord("q")
                elif key_pressed(0x78):  # F9
                    key = ord("q")
            if key in (ord("q"), ord("Q"), 27):
                break
            if config.show_preview and cv2.getWindowProperty(WINDOW_NAME, cv2.WND_PROP_VISIBLE) < 1:
                break

            processing_time = time.perf_counter() - loop_start
            if processing_time < frame_period:
                time.sleep(frame_period - processing_time)
    finally:
        conclusion = build_final_conclusion(
            session_assessments,
            profile.name,
            total_periods=periods_completed,
        )
        recorder.write_conclusion(conclusion)
        if status_overlay is not None:
            status_overlay.close()
        cv2.destroyAllWindows()
    try:
        import tkinter as tk
        from tkinter import messagebox

        dialog = tk.Tk()
        dialog.withdraw()
        messagebox.showinfo(
            "Conclusão da sessão",
            f"{conclusion['title']}\n\n{conclusion['message']}\n\n"
            f"Periodos analisados: {conclusion['cycles']}",
            parent=dialog,
        )
        dialog.destroy()
    except Exception:
        pass
    print(f"[INFO] Monitor encerrado. Resultados: {recorder.session_dir}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Monitor inicial de videoconferencias.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument(
        "--region",
        nargs=4,
        type=int,
        metavar=("LEFT", "TOP", "WIDTH", "HEIGHT"),
        help="Usa uma regiao fixa e ignora a selecao interativa.",
    )
    parser.add_argument(
        "--profile",
        type=Path,
        help="Diretorio de um perfil cadastrado. Sem ele, abre a selecao visual.",
    )
    parser.add_argument(
        "--manual-region",
        action="store_true",
        help="Usa a selecao manual de area em vez da lista de janelas.",
    )
    return parser.parse_args()


def region_from_args(values: list[int] | None) -> CaptureRegion | None:
    if values is None:
        return None
    left, top, width, height = values
    if width <= 0 or height <= 0:
        raise ValueError("A largura e a altura da regiao devem ser positivas.")
    return CaptureRegion(left, top, width, height)


if __name__ == "__main__":
    arguments = parse_args()
    run(
        arguments.config,
        region_from_args(arguments.region),
        arguments.manual_region,
        arguments.profile,
    )
