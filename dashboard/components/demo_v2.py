"""Window Logistic 기반 보이스피싱 위험도 분석 데모.

LLM은 현재 발화에서 원자 이벤트만 추출하고, 위험 판정은 12-01 Feature
Builder와 신뢰된 로컬 PKL의 Window Logistic/Guardrail이 수행한다.
"""

from __future__ import annotations

import json
import os
import re
import unicodedata
from functools import lru_cache
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[2]

try:
    from dotenv import load_dotenv

    # 팀원별 작업 폴더에 둔 비밀키 파일을 명시적으로 읽되 기존 환경변수를 덮어쓰지 않는다.
    load_dotenv(PROJECT_ROOT / "a_함형준" / ".env", override=False)
    load_dotenv(PROJECT_ROOT / ".env", override=False)
except ImportError:
    pass


MODEL_PATH = (
    PROJECT_ROOT
    / "a_함형준"
    / "MLmodel_v2"
    / "WINDOW_LOGISTIC_DASHBOARD_EXPERIMENTAL_SAMPLE_v1.pkl"
)
DASHBOARD_RISK_THRESHOLD = 0.60
WINDOW_TURNS = 10  # 12-01 학습 정의: 현재 Turn을 포함한 최근 최대 10 Turn
WINDOW_STRIDE = 1

EVENT_FAMILIES = ["IMPERSONATION", "PSY_STRATEGY", "ACTION_REQUEST", "MONEY_MOVEMENT", "AMOUNT"]
IMPERSONATION_GROUPS = [
    "PUBLIC_AGENCY", "FINANCIAL_INSTITUTION", "FAMILY", "ACQUAINTANCE",
    "TELECOM_COMPANY", "DELIVERY_LOGISTICS", "OTHER",
]
IMPERSONATION_SUBTYPES = [
    "PROSECUTION", "POLICE", "FSS", "COURT", "POST_OFFICE", "GOVERNMENT_OTHER",
    "BANK", "CARD_COMPANY", "LOAN_COMPANY", "CAPITAL_COMPANY", "SAVINGS_BANK",
    "FINANCIAL_OTHER", "FAMILY", "ACQUAINTANCE", "TELECOM", "DELIVERY", "OTHER",
]
PSY_TYPES = [
    "AUTHORITY", "FEAR", "URGENCY", "LEGITIMACY", "INFO_EXTRACTION", "ISOLATION",
    "MONEY_REQUEST", "BENEFIT", "RESISTANCE_HANDLING", "BEHAVIOR_CONTROL",
]
ACTION_TYPES = [
    "SENSITIVE_INFO", "AUTH_INFO", "DEVICE_CONTROL", "CONTACT_RESTRICTION",
    "CARD_HANDOVER", "ACCOUNT_RENTAL", "OTHER_HIGH_RISK",
]
MONEY_TYPES = [
    "TRANSFER", "WITHDRAWAL", "CASH_HANDOFF", "FEE_PAYMENT", "REPAYMENT",
    "OTHER_MONEY_MOVEMENT",
]

EVENT_OUTPUT_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "events": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "event_family": {"type": "string", "enum": EVENT_FAMILIES},
                    "subtype": {"type": ["string", "null"]},
                    "impersonation_group": {
                        "type": ["string", "null"],
                        "enum": IMPERSONATION_GROUPS + [None],
                    },
                    "evidence_turn_id": {"type": "integer"},
                    "evidence_text": {"type": "string"},
                    "amount_krw": {"type": ["number", "null"]},
                    "amount_context": {"type": ["string", "null"]},
                    "is_requested": {"type": ["boolean", "null"]},
                },
                "required": [
                    "event_family", "subtype", "impersonation_group", "evidence_turn_id",
                    "evidence_text", "amount_krw", "amount_context", "is_requested",
                ],
            },
        }
    },
    "required": ["events"],
}

SYSTEM_INSTRUCTION = """
당신은 금융 통화 텍스트에서 보이스피싱 위험 단서를 원자 Event로 구조화하는 추출기다.
입력에는 과거 CONTEXT Turn과 현재 TARGET Turn 한 개가 있다.

인과 규칙:
- CONTEXT는 TARGET을 이해하기 위한 과거 문맥일 뿐이다.
- TARGET TURN에서 직접 표현되거나 새로 완성된 Event만 출력한다.
- 과거 CONTEXT의 Event를 반복 출력하지 않는다.
- evidence_turn_id는 TARGET TURN 번호와 같아야 한다.
- evidence_text는 TARGET 원문의 실제 연속 구절을 그대로 복사한다.
- 근거가 없거나 애매하면 Event를 만들지 않는다.

IMPERSONATION은 단순한 기관명 언급이 아니라 발화자가 그 기관·신분을 사칭하거나
그 권위를 빌리는 명시적 근거가 있을 때만 추출한다. 정상 상담원의 자기소개를 사칭으로
간주하지 않는다. AMOUNT는 일반 잔액·상품 설명도 추출할 수 있으나, 실제 지급·이동
요구가 아니면 is_requested=false다. MONEY_MOVEMENT와 ACTION_REQUEST는 상대에게
실제 행동을 요구하는 표현이 있을 때만 추출한다.

허용 subtype:
IMPERSONATION: PROSECUTION, POLICE, FSS, COURT, POST_OFFICE, GOVERNMENT_OTHER,
BANK, CARD_COMPANY, LOAN_COMPANY, CAPITAL_COMPANY, SAVINGS_BANK, FINANCIAL_OTHER,
FAMILY, ACQUAINTANCE, TELECOM, DELIVERY, OTHER
PSY_STRATEGY: AUTHORITY, FEAR, URGENCY, LEGITIMACY, INFO_EXTRACTION, ISOLATION,
MONEY_REQUEST, BENEFIT, RESISTANCE_HANDLING, BEHAVIOR_CONTROL
ACTION_REQUEST: SENSITIVE_INFO, AUTH_INFO, DEVICE_CONTROL, CONTACT_RESTRICTION,
CARD_HANDOVER, ACCOUNT_RENTAL, OTHER_HIGH_RISK
MONEY_MOVEMENT: TRANSFER, WITHDRAWAL, CASH_HANDOFF, FEE_PAYMENT, REPAYMENT,
OTHER_MONEY_MOVEMENT

NORMAL/PHISHING, 위험 점수, 최종 판단은 절대 출력하지 않는다.
""".strip()


PSY_SLUG = {
    "AUTHORITY": "authority", "FEAR": "fear", "URGENCY": "urgency",
    "LEGITIMACY": "legitimacy", "INFO_EXTRACTION": "info_extraction",
    "ISOLATION": "isolation", "MONEY_REQUEST": "money_request_strategy",
    "BENEFIT": "benefit", "RESISTANCE_HANDLING": "resistance_handling",
    "BEHAVIOR_CONTROL": "behavior_control",
}
ACTION_SLUG = {
    "SENSITIVE_INFO": "sensitive_info", "AUTH_INFO": "auth_info",
    "DEVICE_CONTROL": "device_control", "CONTACT_RESTRICTION": "contact_restriction",
    "CARD_HANDOVER": "card_handover", "ACCOUNT_RENTAL": "account_rental",
    "OTHER_HIGH_RISK": "other_high_risk_action",
}
MONEY_SLUG = {
    "TRANSFER": "transfer", "WITHDRAWAL": "withdrawal", "CASH_HANDOFF": "cash_handoff",
    "FEE_PAYMENT": "fee_payment", "REPAYMENT": "repayment",
    "OTHER_MONEY_MOVEMENT": "other_money_movement",
}
IMP_GROUP_SLUG = {
    "PUBLIC_AGENCY": "public", "FINANCIAL_INSTITUTION": "financial", "FAMILY": "family",
    "ACQUAINTANCE": "acquaintance", "TELECOM_COMPANY": "telecom",
    "DELIVERY_LOGISTICS": "delivery_logistics", "OTHER": "other",
}
IMP_SUBTYPE_SLUG = {
    "PROSECUTION": "prosecution", "POLICE": "police", "FSS": "fss", "COURT": "court",
    "POST_OFFICE": "post_office", "GOVERNMENT_OTHER": "government_other", "BANK": "bank",
    "CARD_COMPANY": "card_company", "LOAN_COMPANY": "loan_company",
    "CAPITAL_COMPANY": "capital_company", "SAVINGS_BANK": "savings_bank",
    "FINANCIAL_OTHER": "financial_other", "FAMILY": "family_subtype",
    "ACQUAINTANCE": "acquaintance_subtype", "TELECOM": "telecom_subtype",
    "DELIVERY": "delivery_subtype", "OTHER": "other_subtype",
}


@lru_cache(maxsize=1)
def load_window_model() -> dict[str, Any]:
    """프로젝트 내부의 신뢰된 Window bundle을 한 번만 로드한다."""
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"모델 파일을 찾을 수 없습니다: {MODEL_PATH}")
    bundle = joblib.load(MODEL_PATH)
    required = {
        "model", "model_features", "threshold", "guardrail_signal_features",
        "guardrail", "target_mapping", "model_status", "source_run", "model_version",
    }
    missing = sorted(required - set(bundle))
    if missing:
        raise KeyError(f"모델 bundle 필수 항목 누락: {missing}")

    # sklearn 1.6.1에서 저장된 신뢰 모델을 1.8+에서 읽을 때의 호환 필드.
    named_steps = getattr(bundle["model"], "named_steps", {})
    imputer = named_steps.get("imputer")
    if imputer is not None and not hasattr(imputer, "_fill_dtype"):
        fit_dtype = getattr(imputer, "_fit_dtype", None)
        if fit_dtype is not None:
            imputer._fill_dtype = fit_dtype
    return bundle


def _parse_turns(text: str) -> list[str]:
    # 줄바꿈 여부와 관계없이 모든 줄을 문장부호 기준으로 다시 나눈다.
    # 따라서 "첫 문장. 둘째 문장."을 한 줄로 넣어도 두 분석 단위가 된다.
    sentences: list[str] = []
    for line in str(text).splitlines():
        line = line.strip()
        if not line:
            continue
        parts = re.split(r"(?<=[.!?。！？])\s*", line)
        sentences.extend(part.strip() for part in parts if part.strip())
    return sentences


def _prompt_for_target(turns: list[str], target_index: int) -> str:
    # 문장마다 독립적으로 피처화하므로 이전 문장을 CONTEXT에 포함하지 않는다.
    return f"[TARGET][TURN {target_index + 1}][SPEAKER_UNKNOWN] {turns[target_index]}"


def extract_events(text: str) -> dict[str, Any]:
    """각 TARGET Turn을 인과적으로 추출한다. 실패는 0-event와 구분한다."""
    turns = _parse_turns(text)
    if not turns:
        return {"ok": False, "events": [], "turns": [], "error": "빈 발화입니다."}
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        return {
            "ok": False, "events": [], "turns": turns,
            "error": "OPENAI_API_KEY가 설정되지 않아 이벤트 추출을 실행하지 못했습니다.",
        }
    try:
        from openai import OpenAI

        client = OpenAI(api_key=api_key)
        model_name = os.getenv("OPENAI_EVENT_MODEL", "gpt-4o-mini")
        events: list[dict[str, Any]] = []
        successful_turn_ids: list[int] = []
        turn_errors: list[dict[str, Any]] = []
        for target_index, target_text in enumerate(turns):
            try:
                response = client.responses.create(
                    model=model_name,
                    instructions=SYSTEM_INSTRUCTION,
                    input=_prompt_for_target(turns, target_index),
                    text={
                        "format": {
                            "type": "json_schema",
                            "name": "voice_phishing_events_v2_2",
                            "schema": EVENT_OUTPUT_SCHEMA,
                            "strict": True,
                        }
                    },
                )
                parsed = json.loads(response.output_text)
                target_events = parsed.get("events")
                if not isinstance(target_events, list):
                    raise ValueError("events 배열이 없습니다.")
                validated_events = []
                for event in target_events:
                    if int(event["evidence_turn_id"]) != target_index + 1:
                        raise ValueError("evidence_turn_id 불일치")
                    evidence = unicodedata.normalize("NFKC", str(event["evidence_text"]).strip())
                    target_norm = unicodedata.normalize("NFKC", target_text)
                    if not evidence or evidence not in target_norm:
                        raise ValueError("원문과 일치하지 않는 evidence")
                    event["evidence_text"] = evidence
                    event["detected_at_turn"] = target_index + 1
                    validated_events.append(event)
                events.extend(validated_events)
                successful_turn_ids.append(target_index + 1)
            except Exception as exc:
                turn_errors.append({
                    "turn": target_index + 1,
                    "text": target_text,
                    "error": f"{type(exc).__name__}: {exc}",
                })
                continue
        if not successful_turn_ids:
            return {
                "ok": False, "events": [], "turns": turns,
                "successful_turn_ids": [], "turn_errors": turn_errors,
                "error": "모든 문장의 이벤트 추출에 실패했습니다.",
                "extractor_model": model_name,
            }
        return {
            "ok": True, "events": events, "turns": turns, "error": None,
            "successful_turn_ids": successful_turn_ids, "turn_errors": turn_errors,
            "extractor_model": model_name,
        }
    except Exception as exc:
        return {
            "ok": False, "events": [], "turns": turns,
            "error": f"이벤트 추출 실패: {type(exc).__name__}: {exc}",
        }


def _deterministic_amount(text: str, fallback: Any = None) -> float:
    """12-01 원칙에 따라 evidence 원문에서 KRW를 결정적으로 재계산한다."""
    value_text = unicodedata.normalize("NFKC", str(text or "")).replace(",", "")
    total = 0.0
    units = [("억", 100_000_000), ("천만", 10_000_000), ("백만", 1_000_000),
             ("만", 10_000), ("천원", 1_000), ("백원", 100), ("십원", 10)]
    for unit, multiplier in units:
        match = re.search(rf"(\d+(?:\.\d+)?)\s*{unit}", value_text)
        if match:
            total += float(match.group(1)) * multiplier
    if total:
        return total
    match = re.search(r"(\d+(?:\.\d+)?)\s*원", value_text)
    if match:
        return float(match.group(1))
    try:
        return float(fallback) if fallback is not None else np.nan
    except (TypeError, ValueError):
        return np.nan


def _add_repeat_features(out: dict[str, float], prefix: str, count: int) -> None:
    repeat = max(int(count) - 1, 0)
    out[f"{prefix}_present"] = int(count >= 1)
    out[f"{prefix}_repeat"] = repeat
    out[f"{prefix}_accel_tri"] = int(repeat * (repeat - 1) / 2)
    out[f"{prefix}_count_qc"] = int(count)


def _features_from_events(events: list[dict[str, Any]]) -> dict[str, float]:
    """12-01 Feature Builder v2.2를 동일한 이름·계산으로 구현한다."""
    ev = pd.DataFrame(events)
    if ev.empty:
        ev = pd.DataFrame(columns=["event_family", "subtype", "impersonation_group", "amount_krw", "is_requested"])
    out: dict[str, float] = {}

    imp = ev[ev["event_family"].eq("IMPERSONATION")]
    out["imp_present"] = int(len(imp) > 0)
    out["imp_event_count_raw_qc"] = int(len(imp))
    groups = set(imp["impersonation_group"].dropna().astype(str))
    subtypes = set(imp["subtype"].dropna().astype(str))
    for raw, slug in IMP_GROUP_SLUG.items():
        out[f"imp_{slug}"] = int(raw in groups)
    for raw, slug in IMP_SUBTYPE_SLUG.items():
        out[f"imp_{slug}"] = int(raw in subtypes)
    out["imp_group_diversity_qc"] = len(groups)
    out["imp_subtype_diversity_qc"] = len(subtypes)

    psy_types = ev.loc[ev["event_family"].eq("PSY_STRATEGY"), "subtype"].dropna().astype(str).tolist()
    for raw, slug in PSY_SLUG.items():
        _add_repeat_features(out, f"strategy_{slug}", psy_types.count(raw))
    out["strategy_diversity"] = len(set(psy_types))
    out["strategy_event_count_qc"] = len(psy_types)

    action_types = ev.loc[ev["event_family"].eq("ACTION_REQUEST"), "subtype"].dropna().astype(str).tolist()
    for raw, slug in ACTION_SLUG.items():
        _add_repeat_features(out, f"action_{slug}", action_types.count(raw))
    out["action_diversity"] = len(set(action_types))
    out["action_event_count_qc"] = len(action_types)

    money_types = ev.loc[ev["event_family"].eq("MONEY_MOVEMENT"), "subtype"].dropna().astype(str).tolist()
    _add_repeat_features(out, "money_movement", len(money_types))
    for raw, slug in MONEY_SLUG.items():
        _add_repeat_features(out, f"money_{slug}", money_types.count(raw))
    out["money_action_diversity"] = len(set(money_types))
    out["money_event_count_qc"] = len(money_types)

    amount = ev[ev["event_family"].eq("AMOUNT")].copy()
    if not amount.empty:
        amount["amount_krw"] = [
            _deterministic_amount(row.evidence_text, row.amount_krw)
            for row in amount.itertuples()
        ]
        requested = amount[amount["is_requested"].fillna(False).astype(bool)]
    else:
        requested = amount
    req_count = len(requested)
    req_repeat = max(req_count - 1, 0)
    out["amount_mentioned_present"] = int(len(amount) > 0)
    out["amount_event_count_qc"] = int(len(amount))
    out["amount_requested_present"] = int(req_count > 0)
    out["amount_request_repeat"] = req_repeat
    out["amount_request_accel_tri"] = int(req_repeat * (req_repeat - 1) / 2)
    out["amount_request_count_qc"] = int(req_count)
    requested_values = pd.to_numeric(requested.get("amount_krw", pd.Series(dtype=float)), errors="coerce").dropna().clip(lower=0)
    mentioned_values = pd.to_numeric(amount.get("amount_krw", pd.Series(dtype=float)), errors="coerce").dropna().clip(lower=0)
    out["requested_amount_max"] = float(requested_values.max()) if len(requested_values) else 0.0
    out["requested_amount_sum"] = float(requested_values.sum()) if len(requested_values) else 0.0
    out["requested_amount_log1p_max"] = float(np.log1p(out["requested_amount_max"]))
    out["requested_amount_log1p_sum"] = float(np.log1p(out["requested_amount_sum"]))
    out["mentioned_amount_max_qc"] = float(mentioned_values.max()) if len(mentioned_values) else 0.0

    out["signal_family_count"] = int(sum([
        out["imp_present"], int(out["strategy_event_count_qc"] > 0),
        int(out["action_event_count_qc"] > 0), out["money_movement_present"],
        out["amount_requested_present"],
    ]))
    out["ix_imp_authority"] = out["imp_present"] * out["strategy_authority_present"]
    out["ix_public_authority"] = out["imp_public"] * out["strategy_authority_present"]
    out["ix_financial_authority"] = out["imp_financial"] * out["strategy_authority_present"]
    out["ix_authority_urgency"] = out["strategy_authority_present"] * out["strategy_urgency_present"]
    out["ix_fear_urgency"] = out["strategy_fear_present"] * out["strategy_urgency_present"]
    out["ix_info_sensitive"] = out["strategy_info_extraction_present"] * out["action_sensitive_info_present"]
    out["ix_info_auth"] = out["strategy_info_extraction_present"] * out["action_auth_info_present"]
    out["ix_moneystrategy_movement"] = out["strategy_money_request_strategy_present"] * out["money_movement_present"]
    out["ix_fear_money"] = out["strategy_fear_present"] * out["money_movement_present"]
    out["ix_isolation_contact"] = out["strategy_isolation_present"] * out["action_contact_restriction_present"]
    return out


def build_window_features(
    events: list[dict[str, Any]], conversation_context: list[str],
    successful_turn_ids: list[int] | None = None,
) -> list[dict[str, Any]]:
    """각 문장을 독립된 단일 Window로 피처화해 점수 간 희석을 막는다."""
    rows = []
    allowed = set(successful_turn_ids or range(1, len(conversation_context) + 1))
    for end_turn in range(1, len(conversation_context) + 1):
        if end_turn not in allowed:
            continue
        window_events = [
            event for event in events
            if int(event.get("detected_at_turn", event.get("evidence_turn_id", 0))) == end_turn
        ]
        rows.append({
            "start_turn": end_turn,
            "end_turn": end_turn,
            "sentence": conversation_context[end_turn - 1],
            "features": _features_from_events(window_events),
        })
    return rows


def apply_guardrail(raw_probability: float, features: dict[str, float], bundle: dict[str, Any]) -> dict[str, Any]:
    signal_features = list(bundle["guardrail_signal_features"])
    candidate_signal_count = sum(float(features.get(name, 0) or 0) != 0 for name in signal_features)
    threshold = DASHBOARD_RISK_THRESHOLD
    guardrail = bundle["guardrail"]
    final_probability = float(raw_probability)
    applied = candidate_signal_count == 0
    if applied:
        cap_score = float(guardrail.get("zero_feature_cap_score", 20.0)) / 100.0
        margin = float(guardrail.get("threshold_margin", 0.01))
        final_probability = min(final_probability, cap_score, max(0.0, threshold - margin))
    return {
        "raw_ml_risk_score": float(raw_probability) * 100,
        "final_risk_score": final_probability * 100,
        "threshold_score": threshold * 100,
        "candidate_signal_count": int(candidate_signal_count),
        "guardrail_applied": applied,
        "final_label": "PHISHING" if final_probability >= threshold else "NORMAL",
    }


def predict_window(feature_df: pd.DataFrame) -> np.ndarray:
    bundle = load_window_model()
    required = list(bundle["model_features"])
    missing = [name for name in required if name not in feature_df.columns]
    if missing:
        raise ValueError(f"모델 필수 Feature 불일치: {missing}")
    return bundle["model"].predict_proba(feature_df[required])[:, 1]


def get_active_risk_signals(features: dict[str, float]) -> list[dict[str, Any]]:
    labels = {
        "imp_present": "사칭 정황", "strategy_authority_present": "권위 이용",
        "strategy_fear_present": "공포 유발", "strategy_urgency_present": "긴급성 압박",
        "strategy_info_extraction_present": "정보 탐색", "action_sensitive_info_present": "민감정보 요구",
        "action_auth_info_present": "인증정보 요구", "action_contact_restriction_present": "연락 차단 요구",
        "action_other_high_risk_action_present": "기타 고위험 행동 요구",
        "money_movement_present": "금전 이동 요구", "money_transfer_present": "송금 요구",
        "amount_requested_present": "구체적 금액 요구",
    }
    return [
        {"신호": labels[name], "Feature": name, "값": value}
        for name, value in features.items() if name in labels and float(value or 0) > 0
    ]


def analyze_conversation(text: str) -> dict[str, Any]:
    result: dict[str, Any] = {
        "analysis_ok": False, "status": "ANALYSIS_FAILED", "error": None,
        "turns": [], "events": [], "windows": [], "final_label": None,
        "final_risk_score": None, "model_ready": False,
    }
    try:
        bundle = load_window_model()
        result.update({
            "model_ready": True, "model_status": bundle["model_status"],
            "model_version": bundle["model_version"], "source_run": bundle["source_run"],
            "threshold_score": DASHBOARD_RISK_THRESHOLD * 100,
        })
    except Exception as exc:
        result["error"] = f"모델 로드 실패: {type(exc).__name__}: {exc}"
        return result

    extraction = extract_events(text)
    result.update({
        "turns": extraction["turns"], "events": extraction["events"],
        "turn_errors": extraction.get("turn_errors", []),
    })
    if not extraction["ok"]:
        result["error"] = extraction["error"]
        return result

    try:
        windows = build_window_features(
            extraction["events"], extraction["turns"],
            extraction.get("successful_turn_ids"),
        )
        feature_rows = []
        for window in windows:
            row = {name: window["features"].get(name, 0) for name in bundle["model_features"]}
            feature_rows.append(row)
        frame = pd.DataFrame(feature_rows, columns=bundle["model_features"])
        probabilities = predict_window(frame)
        for window, probability in zip(windows, probabilities):
            window.update(apply_guardrail(float(probability), window["features"], bundle))
            window["active_signals"] = get_active_risk_signals(window["features"])

        final_window = windows[-1]
        max_window = max(windows, key=lambda row: row["final_risk_score"])
        evidence = [
            {
                "Turn": int(event["evidence_turn_id"]),
                "유형": f"{event['event_family']} · {event.get('subtype') or '-'}",
                "근거 원문": event["evidence_text"],
            }
            for event in extraction["events"]
        ]
        result.update({
            "analysis_ok": True, "status": "COMPLETED", "windows": windows,
            # 긴 통화에서 짧고 강한 위험 구간이 뒤의 정상 문장에 희석되지 않도록
            # 대표 점수와 최종 판정은 전체 Window의 최고점으로 결정한다.
            "final_label": max_window["final_label"],
            "final_risk_score": max_window["final_risk_score"],
            "raw_ml_risk_score": max_window["raw_ml_risk_score"],
            "guardrail_applied": max_window["guardrail_applied"],
            "candidate_signal_count": max_window["candidate_signal_count"],
            "max_window_score": max_window["final_risk_score"],
            "max_window_start_turn": max_window["start_turn"],
            "max_window_end_turn": max_window["end_turn"],
            "current_window_score": final_window["final_risk_score"],
            "current_window_label": final_window["final_label"],
            "active_signals": get_active_risk_signals(max_window["features"]),
            "evidence": evidence,
            "missing_features": [],
            "unexpected_features": sorted(set(max_window["features"]) - set(bundle["model_features"])),
            "extractor_model": extraction.get("extractor_model"),
            "analyzed_sentence_count": len(windows),
            "failed_sentence_count": len(extraction.get("turn_errors", [])),
        })
        return result
    except Exception as exc:
        result["error"] = f"Feature/모델 추론 실패: {type(exc).__name__}: {exc}"
        return result


EXAMPLES = {
    "보이스피싱 의심 예시": """고객님 명의 계좌가 범죄에 연루됐습니다.
사건번호 확인을 위해 주민등록번호를 말씀해 주세요.
지금부터 누구에게도 말하지 마세요.
안전계좌로 즉시 송금하셔야 합니다.""",
    "정상 금융상담 예시": """안녕하세요. 예금 만기일을 확인하고 싶습니다.
본인인증은 앱에서 직접 진행해 주세요.
만기일은 다음 달 15일입니다.
추가 문의가 있으시면 공식 고객센터로 연락해 주세요.""",
}


def _risk_chart(windows: list[dict[str, Any]], threshold: float) -> go.Figure:
    frame = pd.DataFrame(windows)
    fig = go.Figure(go.Scatter(
        x=frame["end_turn"], y=frame["final_risk_score"], mode="lines+markers",
        line={"color": "#2563eb", "width": 3}, marker={"size": 9},
        customdata=np.stack([frame["start_turn"], frame["raw_ml_risk_score"]], axis=-1),
        hovertemplate="Turn %{customdata[0]}~%{x}<br>최종 점수 %{y:.1f}<br>Raw ML %{customdata[1]:.1f}<extra></extra>",
    ))
    fig.add_hline(y=threshold, line_dash="dash", line_color="#dc2626", annotation_text=f"판정 기준 {threshold:.0f}")
    fig.update_layout(height=330, margin={"l": 25, "r": 20, "t": 25, "b": 25},
                      xaxis_title="현재 발화 Turn", yaxis={"title": "위험점수", "range": [0, 100]})
    return fig


def render_dashboard(result: dict[str, Any]) -> None:
    if not result["analysis_ok"]:
        st.error("분석 실패 / 다시 시도 필요")
        st.caption(result.get("error") or "알 수 없는 오류가 발생했습니다.")
        return

    score = float(result["final_risk_score"])
    highest, current, decision, meta = st.columns([1.2, 1, 1, 1])
    highest.metric("최고 위험점수", f"{score:.1f}점", f"기준 {result['threshold_score']:.0f}점")
    current.metric("마지막 문장 점수", f"{result['current_window_score']:.1f}점", result["current_window_label"])
    decision.metric("최종 상태", result["final_label"])
    meta.metric("최고 구간 위험신호", f"{result['candidate_signal_count']}개")
    st.caption(
        f"대표 위험점수는 문장별 독립 점수 중 최고점이며, 최고 위험 문장은 "
        f"{result['max_window_end_turn']}번 문장이었음."
    )
    if result.get("failed_sentence_count"):
        st.warning(
            f"총 {len(result['turns'])}개 문장 중 {result['failed_sentence_count']}개 문장은 "
            "이벤트 추출에 실패해 점수 계산에서 제외되었습니다."
        )
    if result["final_label"] == "PHISHING":
        st.error("보이스피싱 위험이 높게 감지되었습니다. 송금·정보 제공을 중단하고 공식 채널로 확인하세요.")
    else:
        st.info("현재 판정 기준 미만입니다. NORMAL은 안전 확정을 의미하지 않으므로 계속 주의하세요.")
    if result["guardrail_applied"]:
        st.caption("최고점 Window에도 위험 이벤트가 없어 12-08 Zero-feature Guardrail이 적용되었습니다.")

    st.markdown("#### 문장별 독립 위험점수")
    st.plotly_chart(_risk_chart(result["windows"], result["threshold_score"]), width="stretch", config={"displayModeBar": False})
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("#### 탐지된 주요 위험 신호")
        if result["active_signals"]:
            st.dataframe(pd.DataFrame(result["active_signals"]), hide_index=True, width="stretch")
        else:
            st.success("최고점 Window에서 활성화된 주요 위험 신호가 없습니다.")
    with c2:
        st.markdown("#### 위험 신호의 원문 Evidence")
        if result["evidence"]:
            st.dataframe(pd.DataFrame(result["evidence"]), hide_index=True, width="stretch")
        else:
            st.success("추출된 위험 Event가 없습니다.")

    with st.expander("모델 계산 방식과 검증 정보", expanded=False):
        st.markdown(
            f"""
- LLM은 각 TARGET Turn의 Event만 추출했으며 NORMAL/PHISHING을 판단하지 않았음.
- Python은 **12-01 Feature Builder v2.2** 정의로 피처를 계산했음.
- 입력을 문장 단위로 분리하고 각 문장을 **독립된 단일 Window**로 피처화했음.
- Window Logistic이 ML 확률을 계산한 뒤 **12-08 Zero-feature Guardrail**을 적용했음.
- 대표 위험점수와 최종 판정은 문장별 점수 중 **최고점**을 사용했음.
- 마지막 문장 점수는 별도 지표로 표시했으며, Cumulative와 문장 간 평균은 사용하지 않았음.
- 문장 단위 독립 추론은 긴 입력 대응을 위한 데모 정책이며 학습 당시 연속 Window와 차이가 있어 추가 검증이 필요함.
- 모델 상태: `{result['model_status']}` · 버전: `{result['model_version']}` · source: `{result['source_run']}`
- Event extractor: `{result.get('extractor_model')}`
            """
        )
        st.json({
            "raw_ml_risk_score": result["raw_ml_risk_score"],
            "final_risk_score": result["final_risk_score"],
            "threshold_score": result["threshold_score"],
            "guardrail_applied": result["guardrail_applied"],
            "missing_features": result["missing_features"],
            "unexpected_features": result["unexpected_features"],
        })


def render_demo_v2() -> None:
    st.markdown("## 보이스피싱 위험도 분석 데모 v2")
    st.caption("문장 분리 → 문장별 LLM Event 추출 → Feature Builder → Window Logistic → 최고점 선택 · 판정 기준 60점")
    try:
        bundle = load_window_model()
        if bundle.get("model_status") == "EXPERIMENTAL_SAMPLE":
            st.warning("현재 SAMPLE 기반 실험 모델이며, 실제 서비스 성능을 확정하는 모델이 아닙니다.")
    except Exception as exc:
        st.error(f"모델 준비 실패: {type(exc).__name__}: {exc}")

    selector, load_col = st.columns([3, 1], vertical_alignment="bottom")
    with selector:
        example = st.selectbox("예시 대화", list(EXAMPLES), key="demo_v2_example")
    with load_col:
        if st.button("예시 불러오기", width="stretch", key="demo_v2_load"):
            st.session_state["demo_v2_text"] = EXAMPLES[example]
            st.session_state.pop("demo_v2_result", None)
    text = st.text_area(
        "분석할 통화 텍스트", key="demo_v2_text", height=220,
        placeholder="한 줄에 한 발화를 입력하세요. 화자 표시는 없어도 됩니다.",
    )
    st.caption("입력 문장은 이벤트 추출을 위해 OpenAI API로 전송됩니다. 실제 개인정보는 입력하지 마세요.")
    if st.button("위험도 분석하기", type="primary", width="stretch", key="demo_v2_analyze"):
        if not text.strip():
            st.error("분석할 텍스트를 입력해 주세요.")
        else:
            with st.spinner("문장을 하나씩 분석해 점수를 저장하고 최고 위험점수를 찾고 있습니다..."):
                st.session_state["demo_v2_result"] = analyze_conversation(text)
    if st.session_state.get("demo_v2_result"):
        st.divider()
        render_dashboard(st.session_state["demo_v2_result"])


# pages.py에서 기존 render_demo 이름으로 연결할 수 있는 호환 alias
render_demo = render_demo_v2
