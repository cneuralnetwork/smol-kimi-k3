const el = (id) => document.getElementById(id);
const number = (value, digits = 3) =>
  value === null || value === undefined ? "—" : Number(value).toFixed(digits);

async function refresh() {
  try {
    const response = await fetch(`/api/state?t=${Date.now()}`, { cache: "no-store" });
    const state = await response.json();
    el("status").textContent = state.status || "waiting";
    el("status-dot").className = `status-dot ${state.status || ""}`;
    el("step").textContent =
      state.step === null || state.step === undefined
        ? "—"
        : `${state.step} / ${state.total_steps ?? "?"}`;
    el("train-loss").textContent = number(state.train_loss);
    el("val-loss").textContent = number(state.validation_loss);
    el("throughput").textContent =
      state.tokens_per_second == null ? "—" : Math.round(state.tokens_per_second).toLocaleString();
    el("vram").textContent = number(state.vram_gib, 2);
    el("updated").textContent = state.updated_at
      ? `updated ${new Date(state.updated_at).toLocaleTimeString()}`
      : state.message || "waiting for metrics";
    el("sample-step").textContent =
      state.sample_step == null ? "not generated" : `step ${state.sample_step}`;
    el("sample-prompt").textContent =
      state.sample_prompt ? `PROMPT / ${state.sample_prompt}` : "";
    if (state.sample_text) el("sample").textContent = state.sample_text;
    el("loss-plot").src = `/artifacts/loss.png?t=${Date.now()}`;
  } catch (error) {
    el("status").textContent = "disconnected";
    el("status-dot").className = "status-dot error";
  }
}

refresh();
setInterval(refresh, 2000);
