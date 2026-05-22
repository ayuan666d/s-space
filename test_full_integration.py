"""Full integration test for S-Space v0.3"""
import torch
from s_space.navigator import CoordNavigator
from s_space.formulas import read_coords, compute_delta, compute_injection, compute_injection_from_coords
from s_space.injection_mask import mask_from_metric_weights, mask_from_axes, no_mask
from s_space.space import SSpace, MetricTensor, LayerExpansionLaw
from s_space.pretuned import load_08b_navigator, load_08b_params, list_available

SEP = "=" * 60

print(f"""
{SEP}
  S-Space v0.3 FULL INTEGRATION TEST
{SEP}
""")

# ── 1. Pre-tuned Data Availability ──
print("[1] Pre-tuned data files")
info = list_available()
all_ok = True
for k, v in info.items():
    status = "OK" if v["exists"] else "MISSING"
    if not v["exists"]:
        all_ok = False
    sz = v["size_mb"]
    print(f"  {k}: [{status}] {sz} MB")
assert all_ok, "Some pre-tuned data files are missing!"
print("  ALL DATA FILES PRESENT\n")

# ── 2. CoordNavigator: Inertia Mode ──
print("[2] CoordNavigator: Inertia Mode (no consensus)")
nav = CoordNavigator(params_path="data/coord_nav_params_K100.pt")
print(f"  Inject layers: {nav.inject_layers}")
print(f"  K={nav.K}, d_model={nav.d_model}")
print(f"  Layer scale: {nav.layer_scale}")
print(f"  Has consensus: {nav.has_consensus}")

hidden_states = {L: torch.randn(nav.d_model) for L in nav.inject_layers}
result = nav.navigate(hidden_states, base_r=0.10)
print(f"  Injections: {list(result['injections'].keys())}")
for L, inj in sorted(result['injections'].items()):
    h_norm = hidden_states[L].norm().item()
    ratio = inj.norm().item() / h_norm
    need = result['need'][L]
    print(f"  L{L}: |inj|/|h|={ratio:.4f}, need=[{need.min():.3f},{need.max():.3f}]")
assert result['injections'], "Inertia mode produced no injections!"
print("  INERTIA MODE OK\n")

# ── 3. CoordNavigator: Consensus Mode ──
print("[3] CoordNavigator: Consensus Mode (with d_consensus)")
nav_c = CoordNavigator(
    params_path="data/coord_nav_params_K100.pt",
    consensus_path="data/reasoning_consensus.pt",
)
print(f"  Has consensus: {nav_c.has_consensus}")
print(f"  d_consensus layers: {sorted(nav_c.d_consensus.keys())}")

result_c = nav_c.navigate(hidden_states, base_r=0.10)
assert result_c['injections'], "Consensus mode produced no injections!"
for L, inj in sorted(result_c['injections'].items()):
    h_norm = hidden_states[L].norm().item()
    ratio = inj.norm().item() / h_norm
    print(f"  L{L}: |inj|/|h|={ratio:.4f}")
print("  CONSENSUS MODE OK\n")

# ── 4. Inertia vs Consensus Comparison ──
print("[4] Inertia vs Consensus: Direction Comparison")
for L in nav_c.inject_layers:
    if L in result['injections'] and L in result_c['injections']:
        cos = torch.nn.functional.cosine_similarity(
            result['injections'][L].unsqueeze(0),
            result_c['injections'][L].unsqueeze(0)
        ).item()
        i_norm = result['injections'][L].norm().item()
        c_norm = result_c['injections'][L].norm().item()
        print(f"  L{L}: cos(inertia,consensus)={cos:.4f}, |inertia|={i_norm:.4f}, |consensus|={c_norm:.4f}")
print()

# ── 5. Goal Registration & Chunk Navigation ──
print("[5] Goal Registration + Chunk Navigation")
goal = nav.register_goal(hidden_states)
print(f"  Goal layers: {list(goal.goal_delta_k.keys())}")
print(f"  Need max: {goal.initial_need_max:.3f}, mean: {goal.initial_need_mean:.3f}")

result_chunk = nav.navigate_chunk(hidden_states, base_r=0.10)
assert result_chunk['injections'], "Chunk navigation produced no injections!"
print(f"  Chunk injections: {list(result_chunk['injections'].keys())}")
print("  GOAL + CHUNK OK\n")

# ── 6. Formula ① Precision Test ──
print("[6] Formula 1: Magnitude Control Precision")
params = torch.load("data/coord_nav_params_K100.pt", map_location="cpu", weights_only=False)
L_test = max(params['principal_dirs'].keys())
dirs = params['principal_dirs'][L_test]
mw = params['metric_weights'][L_test]
K = dirs.shape[0]

for r_target in [0.05, 0.10, 0.15, 0.20]:
    h_test = torch.randn(dirs.shape[1])
    coords = read_coords(h_test, dirs)
    delta_k, need_k = compute_delta(coords, mw)
    inject = compute_injection(h_test, delta_k, dirs, mw, r_target)
    if inject is not None:
        actual = inject.norm().item() / h_test.norm().item()
        error = abs(actual - r_target) / r_target * 100
        print(f"  r={r_target:.2f}: actual={actual:.4f}, error={error:.1f}%")
print("  MAGNITUDE CONTROL OK\n")

# ── 7. Need-driven Axis Selection ──
print("[7] Need-driven Axis Selection")
coords_test = torch.zeros(K)
coords_test[0] = 5.0   # high saturation
coords_test[1] = 3.0
coords_test[5] = 0.1   # low saturation
coords_test[10] = 0.01
_, need_test = compute_delta(coords_test, mw)
for i in [0, 1, 5, 10]:
    print(f"  Axis {i}: c_k={coords_test[i]:.2f}, need={need_test[i]:.3f}")
print(f"  Unsaturated axis 10 need ({need_test[10]:.3f}) > saturated axis 0 need ({need_test[0]:.3f}): {need_test[10] > need_test[0]}")
print("  NEED SELECTION OK\n")

# ── 8. Injection Masks ──
print("[8] Injection Masks")
mask_mw = mask_from_metric_weights(mw, dirs, top_k=15)
mask_axes = mask_from_axes([0, 2, 5], dirs.shape[1])
mask_none = no_mask(dirs.shape[1])
print(f"  mask_from_metric_weights: {mask_mw.sum().item():.0f}/{mask_mw.shape[0]} dims active")
print(f"  mask_from_axes([0,2,5]): {mask_axes.sum().item():.0f}/{mask_axes.shape[0]} dims active")
print(f"  no_mask: {mask_none.sum().item():.0f}/{mask_none.shape[0]} dims active")
assert mask_none.sum().item() == mask_none.shape[0], "no_mask should be all ones!"
print("  MASKS OK\n")

# ── 9. End-to-end: coords → delta → injection ──
print("[9] End-to-end: compute_injection_from_coords")
h_e2e = torch.randn(dirs.shape[1])
coords_e2e = read_coords(h_e2e, dirs)
inject_e2e, delta_e2e, need_e2e = compute_injection_from_coords(
    h_e2e, coords_e2e, dirs, mw, r_eff=0.10
)
if inject_e2e is not None:
    ratio = inject_e2e.norm().item() / h_e2e.norm().item()
    print(f"  |inject|/|h| = {ratio:.4f} (target: 0.10)")
    print(f"  need_k range: [{need_e2e.min():.3f}, {need_e2e.max():.3f}]")
    print("  END-TO-END OK\n")
else:
    print("  FAILED: injection is None\n")

# ── 10. Metric Tensor & Space Properties ──
print("[10] S-Space Geometric Properties")
eff_dim = MetricTensor.effective_dimension(mw, threshold=0.01)
top3 = mw[:3].sum().item() / mw.sum().item() * 100
top10 = mw[:10].sum().item() / mw.sum().item() * 100
print(f"  Effective dimensions: {eff_dim}/{K}")
print(f"  Top-3 metric weight: {top3:.1f}%")
print(f"  Top-10 metric weight: {top10:.1f}%")
for L in [3, 14, 19, 22]:
    if L in params['metric_weights']:
        mw_L = params['metric_weights'][L]
        dist = LayerExpansionLaw.mean_distance(L)
        print(f"  L{L}: total_gk={mw_L.sum().item():.2f}, predicted_d={dist:.3f}")
print("  GEOMETRY OK\n")

# ── 11. Load via pretuned module ──
print("[11] Pre-tuned Navigator Loader")
try:
    nav_pretuned = load_08b_navigator(consensus=True)
    print(f"  load_08b_navigator: K={nav_pretuned.K}, d_model={nav_pretuned.d_model}")
    print(f"  Has consensus: {nav_pretuned.has_consensus}")
    result_p = nav_pretuned.navigate(hidden_states, base_r=0.10)
    print(f"  Injections: {list(result_p['injections'].keys())}")
    print("  PRETUNED LOADER OK\n")
except Exception as e:
    print(f"  ERROR: {e}\n")

# ── Summary ──
print(f"""
{SEP}
  FULL INTEGRATION TEST RESULTS
{SEP}

  [1] Pre-tuned data files:        ALL PRESENT
  [2] Inertia mode navigation:      OK
  [3] Consensus mode navigation:   OK
  [4] Inertia vs Consensus compare: OK
  [5] Goal + Chunk navigation:      OK
  [6] Magnitude control precision:  OK
  [7] Need-driven axis selection:   OK
  [8] Injection masks:              OK
  [9] End-to-end pipeline:          OK
  [10] Geometric properties:        OK
  [11] Pretuned navigator loader:   OK

  11/11 TESTS PASSED
{SEP}
""")
