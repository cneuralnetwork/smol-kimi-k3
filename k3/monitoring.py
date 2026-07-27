from __future__ import annotations

import json
import os
import tempfile
from datetime import UTC, datetime
from pathlib import Path

os.environ.setdefault(
    "MPLCONFIGDIR",
    str(Path(tempfile.gettempdir()) / "kimi-k3-matplotlib"),
)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def _now() -> str:
    return datetime.now(UTC).isoformat()


class TrainingMonitor:
    """Durable, dependency-light artifacts consumed by the local dashboard."""

    def __init__(self, run_dir: Path, model_parameters: int, total_steps: int) -> None:
        self.run_dir = run_dir
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.metrics_path = run_dir / "metrics.jsonl"
        self.state_path = run_dir / "dashboard_state.json"
        self.plot_path = run_dir / "loss.png"
        self.train_points: list[tuple[int, float]] = []
        self.validation_points: list[tuple[int, float]] = []
        self.state = {
            "status": "running",
            "updated_at": _now(),
            "model_parameters": model_parameters,
            "total_steps": total_steps,
            "step": None,
            "train_loss": None,
            "validation_loss": None,
            "tokens_per_second": None,
            "vram_gib": None,
            "lr_scale": None,
            "sample_prompt": None,
            "sample_text": None,
            "sample_step": None,
        }
        self._restore_metrics()
        self._write_state()
        self._draw_plot()

    def _restore_metrics(self) -> None:
        if not self.metrics_path.exists():
            return
        for line in self.metrics_path.read_text(encoding="utf-8").splitlines():
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            if record.get("kind") == "train":
                self.train_points.append((record["step"], record["loss"]))
            elif record.get("kind") == "validation":
                self.validation_points.append((record["step"], record["loss"]))

    @staticmethod
    def _atomic_text(path: Path, text: str) -> None:
        temporary = path.with_suffix(path.suffix + ".tmp")
        temporary.write_text(text, encoding="utf-8")
        temporary.replace(path)

    def _write_state(self) -> None:
        self.state["updated_at"] = _now()
        self._atomic_text(self.state_path, json.dumps(self.state, indent=2))

    def _append_metric(self, record: dict) -> None:
        with self.metrics_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record) + "\n")

    def _draw_plot(self) -> None:
        plt.style.use("dark_background")
        figure, axis = plt.subplots(figsize=(9, 4.5), dpi=140)
        figure.patch.set_facecolor("#11100e")
        axis.set_facecolor("#11100e")
        if self.train_points:
            steps, losses = zip(*self.train_points)
            axis.plot(steps, losses, color="#f0d66a", linewidth=1.8, label="train")
        if self.validation_points:
            steps, losses = zip(*self.validation_points)
            axis.plot(
                steps,
                losses,
                color="#70c1b3",
                marker="o",
                markersize=3,
                linewidth=1.2,
                label="validation",
            )
        axis.set_xlabel("optimizer step", color="#aaa49a")
        axis.set_ylabel("cross-entropy loss", color="#aaa49a")
        axis.grid(color="#393631", linewidth=0.6, alpha=0.65)
        axis.tick_params(colors="#aaa49a")
        for spine in axis.spines.values():
            spine.set_color("#393631")
        if self.train_points or self.validation_points:
            axis.legend(frameon=False)
        figure.tight_layout()
        temporary = self.plot_path.with_suffix(".tmp.png")
        figure.savefig(temporary, facecolor=figure.get_facecolor())
        plt.close(figure)
        temporary.replace(self.plot_path)

    def record_train(
        self,
        step: int,
        loss: float,
        lr_scale: float,
        tokens_per_second: float,
        vram_gib: float,
    ) -> None:
        record = {
            "kind": "train",
            "step": step,
            "loss": loss,
            "lr_scale": lr_scale,
            "tokens_per_second": tokens_per_second,
            "vram_gib": vram_gib,
            "timestamp": _now(),
        }
        self._append_metric(record)
        self.train_points.append((step, loss))
        self.state.update(
            step=step,
            train_loss=loss,
            lr_scale=lr_scale,
            tokens_per_second=tokens_per_second,
            vram_gib=vram_gib,
        )
        self._draw_plot()
        self._write_state()

    def record_validation(self, step: int, loss: float) -> None:
        self._append_metric(
            {
                "kind": "validation",
                "step": step,
                "loss": loss,
                "timestamp": _now(),
            }
        )
        self.validation_points.append((step, loss))
        self.state.update(step=step, validation_loss=loss)
        self._draw_plot()
        self._write_state()

    def record_sample(self, step: int, prompt: str, text: str) -> None:
        self.state.update(
            sample_step=step,
            sample_prompt=prompt,
            sample_text=text,
        )
        self._write_state()

    def set_status(self, status: str, message: str | None = None) -> None:
        self.state["status"] = status
        if message is not None:
            self.state["message"] = message
        self._write_state()
