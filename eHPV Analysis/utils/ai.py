import json
import os
from typing import Any, Dict, Iterable, List, Tuple


def _env(name: str) -> str | None:
    val = os.getenv(name)
    return val if val and val.strip() else None


def _is_german(language: str) -> bool:
    return language.lower().startswith("de")


def _safe_float(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _coerce_numeric_map(raw: Dict[str, Any] | None) -> Dict[str, float]:
    result: Dict[str, float] = {}
    if not raw:
        return result
    for key, value in raw.items():
        val = _safe_float(value)
        if val is not None:
            result[str(key)] = val
    return result


def _coerce_text_map(raw: Dict[str, Any] | None) -> Dict[str, str]:
    result: Dict[str, str] = {}
    if not raw:
        return result
    for key, value in raw.items():
        if value is None:
            continue
        text = str(value).strip()
        if text:
            result[str(key)] = text
    return result


def _top_numeric_deltas(
    numeric_delta: Dict[str, float],
    numeric_m1: Dict[str, float],
    numeric_m2: Dict[str, float],
    limit: int = 6,
) -> List[Tuple[str, float, float, float]]:
    rows: List[Tuple[str, float, float, float]] = []
    for col, delta in numeric_delta.items():
        d_val = _safe_float(delta)
        if d_val is None:
            continue
        v1 = numeric_m1.get(col, 0.0)
        v2 = numeric_m2.get(col, 0.0)
        rows.append((col, d_val, v1, v2))
    rows.sort(key=lambda item: abs(item[1]), reverse=True)
    return rows[:limit]


def _structured_fallback(
    m1: str,
    m2: str,
    part: str,
    modulgruppe: str,
    delta_minutes: float,
    language: str,
    context_dict: Dict[str, Any] | None,
    note: str | None = None,
) -> str:
    is_de = _is_german(language)
    text = {
        "analysis": "Analyse fuer" if is_de else "Analysis for",
        "unknown_part": "unbekanntes Bauteil" if is_de else "unknown part",
        "unknown_modul": "unbekannte Modulgruppe" if is_de else "unknown Modulgruppe",
        "minutes": "Minuten" if is_de else "min",
        "drivers": "Treiber" if is_de else "Drivers",
        "process": "Prozess/Werkzeuge" if is_de else "Process/Tooling Evidence",
        "why": "Warum es IEW eHPV beeinflusst" if is_de else "Why it impacts EHPV",
        "actions": "Aktionen" if is_de else "Actions",
        "no_data": "Keine Daten vorhanden." if is_de else "No data available.",
        "model": "Modell" if is_de else "Model",
        "review_delta": "Untersuche" if is_de else "Investigate",
        "protect_delta": "Sichere den Vorteil" if is_de else "Capture the advantage",
        "generic_driver_1": (
            "Vergleiche Befestigungen, Clipse und Betriebsmittel beider Modelle."
            if is_de
            else "Compare fasteners, clips, and tooling between both models."
        ),
        "generic_driver_2": (
            "Analysiere Variantenanteile und Prozessschritte auf Unterschiede."
            if is_de
            else "Review variant content and process steps for differences."
        ),
        "generic_process": (
            "Ergaenze Daten zu Prozessschritten und Werkzeugen, um eine detaillierte Auswertung zu erhalten."
            if is_de
            else "Provide detailed process and tooling data to enrich the analysis."
        ),
        "generic_why": (
            "Ohne Detaildaten wird die Auswirkung auf die IEW eHPV nur allgemein beschrieben."
            if is_de
            else "Without detailed inputs the IEW eHPV impact can only be described in general terms."
        ),
        "generic_action": (
            "Priorisiere das Sammeln strukturierter Daten zu Schrauben, Clipse und Betriebsmitteln."
            if is_de
            else "Capture structured data for screws, clips, and tooling to prioritise improvements."
        ),
        "note_prefix": (
            "Hinweis (Fallback):" if is_de else "Note (Fallback):"
        ),
    }

    part_label = part or text["unknown_part"]
    modul_label = modulgruppe or text["unknown_modul"]

    lines: List[str] = [
        f"{text['analysis']}: {part_label}",
        f"Modulgruppe: {modul_label}",
        f"Delta IEW eHPV: {delta_minutes:+.2f} {text['minutes']}",
        "",
        f"[{text['drivers']}]",
    ]

    drivers: List[str] = []
    process_lines: List[str] = []
    why_lines: List[str] = []
    action_lines: List[str] = []

    model1_info = context_dict.get("model_1") if isinstance(context_dict, dict) else {}
    model2_info = context_dict.get("model_2") if isinstance(context_dict, dict) else {}
    model1_name = str(model1_info.get("name") or m1)
    model2_name = str(model2_info.get("name") or m2)
    model_label_1 = f"{text['model']} {model1_name}"
    model_label_2 = f"{text['model']} {model2_name}"

    numeric_m1 = _coerce_numeric_map(model1_info.get("numeric_sums") if isinstance(model1_info, dict) else {})
    numeric_m2 = _coerce_numeric_map(model2_info.get("numeric_sums") if isinstance(model2_info, dict) else {})
    numeric_delta = _coerce_numeric_map(context_dict.get("numeric_delta") if isinstance(context_dict, dict) else {})

    top_deltas = _top_numeric_deltas(numeric_delta, numeric_m1, numeric_m2)

    if top_deltas:
        for col, delta_val, val1, val2 in top_deltas:
            drivers.append(
                f"- {col}: {model2_name} {val2:.2f} vs {model1_name} {val1:.2f} (Delta {delta_val:+.2f})."
            )
    else:
        drivers.extend(
            [
                f"- {text['generic_driver_1']}",
                f"- {text['generic_driver_2']}",
            ]
        )

    lines.extend(drivers or [f"- {text['no_data']}"])
    lines.append("")
    lines.append(f"[{text['process']}]")

    attrs_m1 = _coerce_text_map(model1_info.get("attributes") if isinstance(model1_info, dict) else {})
    attrs_m2 = _coerce_text_map(model2_info.get("attributes") if isinstance(model2_info, dict) else {})

    if attrs_m1 or attrs_m2:
        seen_cols: Iterable[str] = sorted({*attrs_m1.keys(), *attrs_m2.keys()})[:6]
        for col in seen_cols:
            val1 = attrs_m1.get(col)
            val2 = attrs_m2.get(col)
            if val1:
                process_lines.append(f"- {model_label_1}: {col} = {val1}")
            if val2:
                process_lines.append(f"- {model_label_2}: {col} = {val2}")
    else:
        process_lines.append(f"- {text['generic_process']}")

    lines.extend(process_lines or [f"- {text['no_data']}"])
    lines.append("")
    lines.append(f"[{text['why']}]")

    value_m1 = _safe_float(model1_info.get("iew_ehpv")) if isinstance(model1_info, dict) else None
    value_m2 = _safe_float(model2_info.get("iew_ehpv")) if isinstance(model2_info, dict) else None
    if value_m1 is not None and value_m2 is not None:
        why_lines.append(
            f"- {model2_name} IEW eHPV {value_m2:.2f} vs {model1_name} {value_m1:.2f} "
            f"(Delta {delta_minutes:+.2f} {text['minutes']})."
        )
    if top_deltas:
        col, delta_val, val1, val2 = top_deltas[0]
        why_lines.append(
            f"- {col} difference of {delta_val:+.2f} "
            f"({model2_name} {val2:.2f} vs {model1_name} {val1:.2f}) drives the delta."
        )
    if not why_lines:
        why_lines.append(f"- {text['generic_why']}")

    lines.extend(why_lines)
    lines.append("")
    lines.append(f"[{text['actions']}]")

    if top_deltas:
        for col, delta_val, val1, val2 in top_deltas[:3]:
            if delta_val >= 0:
                action_lines.append(
                    f"- {text['review_delta']} {col} bei {model2_name}: "
                    f"{model2_name} {val2:.2f} vs {model1_name} {val1:.2f} (Delta {delta_val:+.2f})."
                    if is_de
                    else f"- {text['review_delta']} how to reduce {col} for {model2_name}: "
                    f"{model2_name} {val2:.2f} vs {model1_name} {val1:.2f} (Delta {delta_val:+.2f})."
                )
            else:
                action_lines.append(
                    f"- {text['protect_delta']} bei {col} und uebertrage auf {model1_name}: "
                    f"{model2_name} {val2:.2f} vs {model1_name} {val1:.2f} (Delta {delta_val:+.2f})."
                    if is_de
                    else f"- {text['protect_delta']} in {col} and transfer to {model1_name}: "
                    f"{model2_name} {val2:.2f} vs {model1_name} {val1:.2f} (Delta {delta_val:+.2f})."
                )
    else:
        action_lines.append(f"- {text['generic_action']}")

    lines.extend(action_lines)

    if note:
        lines.extend(["", f"{text['note_prefix']} {note}"])

    return "\n".join(lines)


def try_azure_openai_summary(
    m1,
    m2,
    part,
    modulgruppe,
    delta_minutes,
    language,
    context,
):
    endpoint = _env("AZURE_OPENAI_ENDPOINT")
    api_key = _env("AZURE_OPENAI_API_KEY")
    dep_name = _env("AZURE_OPENAI_DEPLOYMENT")

    context_dict: Dict[str, Any] | None = None
    if isinstance(context, (dict, list)):
        context_dict = context if isinstance(context, dict) else {"data": context}
        context_for_prompt = json.dumps(context, ensure_ascii=False, indent=2)
    else:
        context_for_prompt = str(context)
        try:
            context_dict = json.loads(context_for_prompt)
        except Exception:
            context_dict = None

    if not (endpoint and api_key and dep_name):
        return _structured_fallback(
            m1,
            m2,
            part,
            modulgruppe,
            delta_minutes,
            language,
            context_dict,
            note="Azure OpenAI configuration missing.",
        )

    try:
        from openai import AzureOpenAI

        client = AzureOpenAI(
            azure_endpoint=endpoint,
            api_key=api_key,
            api_version="2024-05-01-preview",
        )
        prompt = f"""
whats weather today
"""
        resp = client.chat.completions.create(
            model=dep_name,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
            max_tokens=400,
        )
        return resp.choices[0].message.content
    except Exception as exc:
        return _structured_fallback(
            m1,
            m2,
            part,
            modulgruppe,
            delta_minutes,
            language,
            context_dict,
            note=f"Azure OpenAI error: {exc}",
        )
