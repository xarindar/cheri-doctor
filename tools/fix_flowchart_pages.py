"""Fix diagnostic flowchart pages in the vision cache.

Pages 397 and 411 were extracted by Gemini as fragmented procedure blocks
instead of structured flowchart figures. This replaces their cache entries
with properly structured flowchart representations.
"""

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
from src.utils import load_json, save_json

cache_path = PROJECT_ROOT / "build" / "full_page_vision_cache.json"

with open(cache_path, encoding="utf-8") as f:
    cache = json.load(f)

# Page 397: CODE 14 - Coolant Temperature Sensor diagnostic flowchart
cache["397"] = {
    "page": 397,
    "section_id": "6E2-A",
    "section_title": "DRIVEABILITY AND EMISSIONS 1.0L (VIN 6)",
    "page_label": "6E2-A-37",
    "reading_order": ["block_0", "block_1", "block_2", "block_3"],
    "blocks": [
        {
            "block_id": "block_0",
            "type": "header",
            "title": "CODE 14 COOLANT TEMPERATURE SENSOR (CTS) CIRCUIT (LOW TEMPERATURE INDICATED) 1.0L (VIN 6) \"M\" CARLINE (TBI)",
            "procedure_type": None,
            "text": None,
            "steps": [],
            "rows": [],
            "caption": None,
            "legend": {},
            "associated_figure_ids": [],
            "continues_from_previous_page": False,
            "continues_to_next_page": False,
        },
        {
            "block_id": "block_1",
            "type": "note",
            "title": None,
            "procedure_type": None,
            "text": "IF CODE 22 IS PRESENT, REPAIR THAT CODE FIRST AND RECHECK FOR ADDITIONAL CODES.",
            "steps": [],
            "rows": [],
            "caption": None,
            "legend": {},
            "associated_figure_ids": [],
            "continues_from_previous_page": False,
            "continues_to_next_page": False,
        },
        {
            "block_id": "block_2",
            "type": "figure",
            "title": "Code 14 Coolant Temperature Sensor (CTS) Circuit Diagnostic Flowchart",
            "procedure_type": None,
            "text": (
                'step_1: IGNITION "OFF," CLEAR CODES. DIAGNOSIS SWITCH TERMINAL NOT GROUNDED. '
                'START ENGINE AND RUN FOR 1 MINUTE OR UNTIL "CHECK ENGINE" LIGHT COMES "ON." '
                'IGNITION "ON," ENGINE STOPPED. GROUND DIAGNOSIS SWITCH TERMINAL AND NOTE CODE.\n'
                "step_1_code_14: Proceed to Step 2\n"
                "step_1_no_code_14: PROBLEM IS INTERMITTENT. IF NO OTHER CODES WERE STORED, "
                'REFER TO "DIAGNOSTICS AIDS" ON FACING PAGE OR "INTERMITTENTS," SECTION "B".\n'
                'step_2: IGNITION "OFF," CLEAR CODES. DISCONNECT COOLANT TEMPERATURE SENSOR CONNECTOR. '
                "JUMPER HARNESS TERMINALS TOGETHER. START ENGINE AND RUN FOR 1 MINUTE OR UNTIL "
                '"CHECK ENGINE" LIGHT COMES "ON." IGNITION "ON," ENGINE STOPPED. '
                "GROUND DIAGNOSIS SWITCH TERMINAL AND NOTE CODE.\n"
                "step_2_code_14: Proceed to voltage check\n"
                "step_2_code_15: FAULTY COOLANT SENSOR CONNECTION OR FAULTY SENSOR.\n"
                'voltage_check: IGNITION "ON," ENGINE STOPPED. PROBE COOLANT TEMPERATURE SENSOR '
                "HARNESS GRY/WHT WIRE WITH A VOLTMETER TO GROUND. SHOULD BE 4-6 VOLTS.\n"
                "voltage_ok: SENSOR GROUND CIRCUIT OPEN OR FAULTY ECM CONNECTION OR ECM.\n"
                "voltage_not_ok: GRY/WHT WIRE OPEN OR FAULTY ECM CONNECTION OR ECM."
            ),
            "steps": [],
            "rows": [],
            "caption": None,
            "legend": {},
            "associated_figure_ids": [],
            "continues_from_previous_page": False,
            "continues_to_next_page": False,
        },
        {
            "block_id": "block_3",
            "type": "notice",
            "title": None,
            "procedure_type": None,
            "text": 'CLEAR CODES AND CONFIRM NO "CHECK ENGINE" LIGHT.',
            "steps": [],
            "rows": [],
            "caption": None,
            "legend": {},
            "associated_figure_ids": [],
            "continues_from_previous_page": False,
            "continues_to_next_page": False,
        },
    ],
}

# Page 411: CODE 25 - MAT Sensor diagnostic flowchart
cache["411"] = {
    "page": 411,
    "section_id": "6E2-A",
    "section_title": "DRIVEABILITY AND EMISSIONS 1.0L (VIN 6)",
    "page_label": "6E2-A-51",
    "reading_order": ["block_0", "block_1", "block_2", "block_3"],
    "blocks": [
        {
            "block_id": "block_0",
            "type": "header",
            "title": "CODE 25 MANIFOLD AIR TEMPERATURE (MAT) SENSOR CIRCUIT (HIGH TEMPERATURE INDICATED) 1.0L (VIN 6) \"M\" CARLINE (TBI)",
            "procedure_type": None,
            "text": None,
            "steps": [],
            "rows": [],
            "caption": None,
            "legend": {},
            "associated_figure_ids": [],
            "continues_from_previous_page": False,
            "continues_to_next_page": False,
        },
        {
            "block_id": "block_1",
            "type": "figure",
            "title": "Code 25 MAT Sensor Circuit Diagnostic Flowchart",
            "procedure_type": None,
            "text": (
                'step_1: DISCONNECT MAT SENSOR. IGNITION "ON", ENGINE STOPPED. '
                "CHECK VOLTAGE BETWEEN AIR TEMPERATURE SENSOR HARNESS CONNECTOR "
                "TERMINALS USING A DIGITAL VOLTMETER (J 34029-1) OR EQUIVALENT.\n"
                "step_1_4v_or_over: Proceed to Step 2\n"
                "step_1_below_4v: GRAY WIRE SHORTED TO GROUND OR GRAY WIRE SHORTED "
                "TO SENSOR GROUND CIRCUIT OR FAULTY ECM\n"
                "step_2: CHECK RESISTANCE ACROSS MAT SENSOR TERMINALS. SHOULD BE MORE "
                "THAN 185 OHMS WITH WARM ENGINE. SEE TABLE FOR APPROXIMATE TEMPERATURE "
                "TO RESISTANCE VALUES.\n"
                "step_2_ok: INTERMITTENT FAULT IN SENSOR CIRCUIT OR CONNECTOR. IF ADDITIONAL "
                'CODES WERE STORED, USE APPLICABLE CHART. IF NO CODES, REFER TO "INTERMITTENTS", '
                'SECTION "B".\n'
                "step_2_not_ok: REPLACE SENSOR"
            ),
            "steps": [],
            "rows": [],
            "caption": None,
            "legend": {},
            "associated_figure_ids": [],
            "continues_from_previous_page": False,
            "continues_to_next_page": False,
        },
        {
            "block_id": "block_2",
            "type": "table",
            "title": "DIAGNOSTIC AID MAT SENSOR TEMPERATURE TO RESISTANCE VALUES (APPROXIMATE)",
            "procedure_type": None,
            "text": None,
            "steps": [],
            "rows": [
                ["\u00b0F", "\u00b0C", "OHMS"],
                ["210", "100", "185"],
                ["160", "70", "450"],
                ["100", "38", "1,800"],
                ["70", "20", "3,400"],
                ["40", "4", "7,500"],
                ["20", "-7", "13,500"],
                ["0", "-18", "25,000"],
                ["-40", "-40", "100,700"],
            ],
            "caption": None,
            "legend": {},
            "associated_figure_ids": [],
            "continues_from_previous_page": False,
            "continues_to_next_page": False,
        },
        {
            "block_id": "block_3",
            "type": "notice",
            "title": None,
            "procedure_type": None,
            "text": 'CLEAR CODES AND CONFIRM NO "CHECK ENGINE" LIGHT.',
            "steps": [],
            "rows": [],
            "caption": None,
            "legend": {},
            "associated_figure_ids": [],
            "continues_from_previous_page": False,
            "continues_to_next_page": False,
        },
    ],
}

with open(cache_path, "w", encoding="utf-8") as f:
    json.dump(cache, f, indent=2, ensure_ascii=False)

print(f"Fixed pages 397 and 411 as flowchart figures")
print(f"Cache size: {len(cache)} pages")
