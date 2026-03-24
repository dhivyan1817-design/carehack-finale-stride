import numpy as np


def _etc_text(etc):
    """Human-readable ETC string — never outputs 'inf'."""
    if etc == 0.0:
        return "already critical"
    if etc == np.inf or etc > 9000:
        return "no imminent failure projected"
    return f"{etc:.0f} hrs"


def classify_urgency(row):
    """
    Explainable logic to convert ML outputs into urgency categories.
    Zone-centric: evaluates structural zones, not individual cracks.

    Key principle:
      - SRI reflects accumulated damage (can stay high even while stabilizing)
      - DPR reflects current trend (negative = improving right now)
      - Both must be considered together for a coherent assessment
    """
    sri = row['sri']
    dpr = row['dpr']
    acceleration = row['damage_acceleration']
    etc = row['etc']
    stabilizing = dpr <= 0

    # ── Priority 1: Immediate Attention ──────────────────────────────────────
    if sri >= 0.8:
        if stabilizing:
            return (
                "Immediate Attention",
                f"Zone has sustained critical damage (SRI: {sri:.3f}). "
                f"Although progression is currently slowing (DPR: {dpr:.5f}), "
                f"the structural risk level remains dangerous. Immediate inspection required."
            )
        return (
            "Immediate Attention",
            f"Zone SRI is critical ({sri:.3f} ≥ 0.8) and still rising. "
            f"Estimated time to structural failure threshold: {_etc_text(etc)}."
        )

    if acceleration > 0.05:
        return (
            "Immediate Attention",
            f"Damage is accelerating rapidly in this zone (acceleration index: {acceleration:.4f}). "
            f"Escalating risk — do not wait for SRI to cross threshold."
        )

    # ── Priority 2: Repair Soon ───────────────────────────────────────────────
    if sri > 0.5:
        if stabilizing:
            return (
                "Repair Soon",
                f"Zone has elevated accumulated damage (SRI: {sri:.3f}), "
                f"but progression is currently stabilizing (DPR: {dpr:.5f}). "
                f"Schedule a structural inspection — condition could deteriorate again."
            )
        return (
            "Repair Soon",
            f"Zone SRI is elevated ({sri:.3f} > 0.50) and progressing. "
            f"Estimated time to critical threshold: {_etc_text(etc)}."
        )

    if dpr > 0.02:
        return (
            "Repair Soon",
            f"High Damage Progression Rate ({dpr:.5f}/hr) detected. "
            f"Zone is deteriorating faster than safe limits — {_etc_text(etc)}."
        )

    # ── Priority 3: Monitor ───────────────────────────────────────────────────
    if stabilizing and sri > 0.1:
        return (
            "Monitor",
            f"Zone risk is stabilizing (DPR: {dpr:.5f}). "
            f"Current SRI ({sri:.3f}) remains above baseline — continue routine monitoring."
        )

    return "Monitor", f"Zone condition is stable (SRI: {sri:.3f}). No immediate action required."


def apply_decision_logic(df):
    """
    Applies urgency classification and explanations to the dataframe.
    """
    results = df.apply(classify_urgency, axis=1)
    df['urgency_category'] = [r[0] for r in results]
    df['urgency_explanation'] = [r[1] for r in results]
    return df
