"""Six incident templates.

Each template realises a single base transcript whose turns carry ground-truth
annotations.  The generator combines the base with one of five difficulty
conditions (clean, chaotic, multi_speaker, code_switched, extended_interview)
to yield 6x5 = 30 base transcripts.

Each turn is a dict:
    {
        "speaker": "OFFICER" | "SUSPECT" | "COMPLAINANT" | "WITNESS_k" | "DISPATCH",
        "text":    str,
        "tags":    list[str]   # subset of {"evid", "incul", "excul"}
    }

``entities`` lists the named entities (people, places, vehicles, identifiers,
dates) ground-truth-present in the transcript -- used by NEHR.
"""
from __future__ import annotations

from copy import deepcopy
from typing import Iterable

CATEGORIES = ("domestic_violence", "drug", "financial", "dui", "property", "traffic")
CONDITIONS = ("clean", "chaotic", "multi_speaker", "code_switched", "extended_interview")


# ---------------------------------------------------------------------------
# Base templates
# ---------------------------------------------------------------------------

def _t(speaker: str, text: str, *tags: str) -> dict:
    return {"speaker": speaker, "text": text, "tags": list(tags)}


def _dv() -> dict:
    return {
        "category": "domestic_violence",
        "entities": [
            "Maria Hernandez", "Carlos Hernandez", "Officer Reyes",
            "1428 Maple Avenue", "Apartment 3B", "March 14, 2026",
        ],
        "turns": [
            _t("DISPATCH", "Unit 24, respond to 1428 Maple Avenue, Apartment 3B, "
                          "reports of a domestic disturbance, possible physical altercation."),
            _t("OFFICER", "Copy dispatch, en route to 1428 Maple Avenue."),
            _t("OFFICER", "Ma'am, this is Officer Reyes, can you open the door?"),
            _t("COMPLAINANT", "He hit me in the face.", "evid", "incul"),
            _t("COMPLAINANT", "My husband, Carlos. He grabbed my arm and pushed me into the wall.",
                "incul"),
            _t("OFFICER", "Is he still inside the apartment?"),
            _t("COMPLAINANT", "He left about ten minutes ago in his blue Honda."),
            _t("OFFICER", "I see redness on your left cheek and a small cut on your lip. "
                         "Do you need medical attention?"),
            _t("COMPLAINANT", "No, I just want to make a report."),
            _t("COMPLAINANT", "He told me, if I called the police, he would come back and finish it.",
                "evid", "incul"),
            _t("WITNESS_1", "I'm her neighbor. I heard shouting around eight and then a thud.",
                "incul"),
            _t("WITNESS_1", "But I didn't actually see anything through the door.", "excul"),
            _t("COMPLAINANT", "Honestly, he was drinking, and I yelled at him first.", "excul"),
            _t("COMPLAINANT", "He has never hit me before tonight.", "excul"),
            _t("OFFICER", "Has he been drinking? Do you know if he uses any substances?"),
            _t("COMPLAINANT", "Just beer tonight. He took two Vicodin this afternoon for his back."),
            _t("OFFICER", "Thank you, ma'am. I will document this and request a "
                         "follow-up by the domestic-violence unit."),
        ],
    }


def _drug() -> dict:
    return {
        "category": "drug",
        "entities": [
            "Jamal Carter", "Officer Patel", "Interstate 5", "Exit 142",
            "2017 Toyota Camry", "License 7XJK329", "April 2, 2026",
        ],
        "turns": [
            _t("OFFICER", "I clocked the silver Camry doing seventy-eight in a sixty-five "
                         "zone on Interstate 5 near Exit 142."),
            _t("OFFICER", "Driver, license and registration please."),
            _t("SUSPECT", "Yeah, here. My name is Jamal Carter."),
            _t("OFFICER", "Sir, do you know why I pulled you over?"),
            _t("SUSPECT", "I figured I was going a little fast.", "incul"),
            _t("OFFICER", "I smell something coming from the vehicle. Have you been smoking?"),
            _t("SUSPECT", "No sir, I have not smoked anything today.", "excul"),
            _t("OFFICER", "There is a small plastic bag on your passenger seat. What is in it?"),
            _t("SUSPECT", "That is not mine. My cousin borrowed the car last night.", "excul"),
            _t("OFFICER", "I am going to have you step out of the vehicle. "
                         "Keep your hands where I can see them."),
            _t("OFFICER", "Based on the visible plastic baggie and the odor of cannabis, "
                         "I am going to conduct a probable-cause search of the vehicle."),
            _t("OFFICER", "I located a sealed bag containing what appears to be approximately "
                         "four grams of methamphetamine in the center console."),
            _t("SUSPECT", "I swear that is not mine. I had no idea that was in there.", "excul", "evid"),
            _t("OFFICER", "I am placing you under arrest for possession of a controlled substance. "
                         "You have the right to remain silent."),
            _t("SUSPECT", "I understand my rights. I want a lawyer.", "evid"),
        ],
    }


def _financial() -> dict:
    return {
        "category": "financial",
        "entities": [
            "Priya Shah", "Officer Donnelly", "First National Bank",
            "Branch 217", "Check 4982", "May 18, 2026", "$2,400",
        ],
        "turns": [
            _t("DISPATCH", "Unit 12, respond to First National Bank Branch 217, "
                          "report of an attempted forged-check transaction."),
            _t("WITNESS_1", "I am Priya Shah, the bank teller. A man tried to cash a check "
                            "for two thousand four hundred dollars."),
            _t("WITNESS_1", "The signature did not match our records on file for the account holder.",
                "incul"),
            _t("OFFICER", "Did he leave the building?"),
            _t("WITNESS_1", "No, he is sitting in the lobby. He says it's his uncle's check."),
            _t("OFFICER", "Sir, can you tell me how you came into possession of this check?"),
            _t("SUSPECT", "My uncle gave it to me to cover rent. His name is on it.", "excul"),
            _t("SUSPECT", "I did not forge anything.", "excul", "evid"),
            _t("OFFICER", "What is your uncle's name and phone number?"),
            _t("SUSPECT", "I do not remember the number. He just told me to deposit it."),
            _t("WITNESS_1", "I called the account holder before flagging this. "
                            "He said he never wrote this check and the checkbook was stolen "
                            "two weeks ago.", "incul", "evid"),
            _t("OFFICER", "Did you sign the back of this check yourself, sir?"),
            _t("SUSPECT", "Yes, I endorsed it.", "incul"),
            _t("OFFICER", "Based on the account holder's statement and your admission of endorsement, "
                         "I have probable cause to detain you for forgery and uttering. "
                         "Stand up and turn around please."),
        ],
    }


def _dui() -> dict:
    return {
        "category": "dui",
        "entities": [
            "Sarah Kim", "Officer Walker", "Main Street", "Pine Avenue",
            "2019 Subaru Outback", "License 5BHT902", "August 11, 2026",
        ],
        "turns": [
            _t("OFFICER", "I observed the Subaru cross the double-yellow line twice "
                         "on Main Street between Oak and Pine Avenue at approximately 11:42 p.m."),
            _t("OFFICER", "Driver, license and registration. Have you been drinking tonight?"),
            _t("SUSPECT", "I had two glasses of wine with dinner around eight.", "incul"),
            _t("OFFICER", "Where were you coming from?"),
            _t("SUSPECT", "From Bertolini's on Pine Avenue. I am Sarah Kim."),
            _t("OFFICER", "I smell an odor of alcoholic beverage emanating from your breath. "
                         "Your eyes appear watery."),
            _t("SUSPECT", "I take prescription Lexapro. The label warns about drowsiness.", "excul"),
            _t("OFFICER", "Please step out of the vehicle. I'm going to administer "
                         "field sobriety tests."),
            _t("OFFICER", "On the horizontal gaze nystagmus test I observed sustained nystagmus "
                         "at maximum deviation in both eyes."),
            _t("OFFICER", "On the walk-and-turn I observed three out of eight clues. "
                         "On the one-leg-stand I observed two out of four clues."),
            _t("SUSPECT", "I have a bad ankle from a soccer injury last year.", "excul", "evid"),
            _t("OFFICER", "I am going to ask you to provide a breath sample on the "
                         "preliminary alcohol screening device."),
            _t("OFFICER", "The PAS device reads 0.09 percent. I am placing you under arrest "
                         "for driving under the influence."),
            _t("SUSPECT", "I am completely cooperative. Please do not impound my car.", "excul"),
        ],
    }


def _property() -> dict:
    return {
        "category": "property",
        "entities": [
            "David Chen", "Marcus Reilly", "Officer Nguyen", "552 Birch Lane",
            "Sony PlayStation 5", "MacBook Pro", "October 7, 2026",
        ],
        "turns": [
            _t("DISPATCH", "Unit 18, respond to 552 Birch Lane, reported residential burglary, "
                          "homeowner is on scene, suspect possibly still in the area."),
            _t("WITNESS_1", "I am David Chen, the homeowner. I came home from work and "
                            "the rear sliding door was pried open."),
            _t("WITNESS_1", "My PlayStation 5 and MacBook Pro are missing. The MacBook has a "
                            "blue sticker of a cat on the lid."),
            _t("OFFICER", "When did you leave the residence this morning?"),
            _t("WITNESS_1", "I left at seven a.m. for work."),
            _t("WITNESS_2", "I live next door. Around two p.m. I saw a man in a red sweatshirt "
                            "walking quickly out of the back gate carrying a backpack.", "incul"),
            _t("WITNESS_2", "I could not see his face clearly because he had the hood up.",
                "excul"),
            _t("OFFICER", "Approximately two blocks east, I observed a male, later identified "
                         "as Marcus Reilly, matching that description sitting on a bench "
                         "with a black backpack."),
            _t("OFFICER", "Sir, what is in the backpack?"),
            _t("SUSPECT", "Just my own stuff. Nothing in there is stolen.", "excul"),
            _t("OFFICER", "Do I have your consent to look in the backpack?"),
            _t("SUSPECT", "I don't really want you searching it.", "excul", "evid"),
            _t("OFFICER", "Based on the witness description, the timing, and your proximity "
                         "to the scene I am detaining you for investigation."),
            _t("OFFICER", "Through the unzipped main compartment of the backpack I observed "
                         "a MacBook Pro with a blue cat sticker on the lid in plain view.",
                "incul"),
        ],
    }


def _traffic() -> dict:
    return {
        "category": "traffic",
        "entities": [
            "Robert Johnson", "Officer Mendez", "Elm Street", "Cedar Avenue",
            "2014 Ford F-150", "License 9KFM451", "December 3, 2026",
        ],
        "turns": [
            _t("OFFICER", "I observed a white Ford F-150 fail to stop at the posted stop sign "
                         "at the intersection of Elm and Cedar at approximately 4:18 p.m."),
            _t("OFFICER", "I activated my overhead lights and the vehicle yielded approximately "
                         "fifty yards south of the intersection."),
            _t("OFFICER", "Sir, license and registration. Do you know why I pulled you over?"),
            _t("SUSPECT", "I came to a complete stop. I slowed down to less than five miles per hour.",
                "excul", "evid"),
            _t("OFFICER", "From my position at the southwest corner of the intersection I "
                         "observed your vehicle continue through at approximately fifteen miles "
                         "per hour without stopping."),
            _t("SUSPECT", "I'm Robert Johnson. I drive this route every day. There was a delivery "
                          "truck blocking your view from where you were parked.", "excul"),
            _t("WITNESS_1", "I'm his wife in the passenger seat. He absolutely stopped, "
                            "I felt the brake.", "excul", "evid"),
            _t("OFFICER", "Ma'am, I am going to ask you to remain quiet while I speak to the driver."),
            _t("OFFICER", "Sir, your registration tag expired last month."),
            _t("SUSPECT", "I have the renewal paperwork right here, I paid online two weeks ago.",
                "excul"),
            _t("OFFICER", "I am going to issue you a citation for failure to stop. "
                         "The expired registration is fixable and I will issue a warning."),
            _t("SUSPECT", "I am going to contest this in court.", "excul"),
        ],
    }


_BASE_BUILDERS = {
    "domestic_violence": _dv,
    "drug":              _drug,
    "financial":         _financial,
    "dui":               _dui,
    "property":          _property,
    "traffic":           _traffic,
}


def base_template(category: str) -> dict:
    if category not in _BASE_BUILDERS:
        raise KeyError(f"Unknown category: {category}")
    return deepcopy(_BASE_BUILDERS[category]())


# ---------------------------------------------------------------------------
# Difficulty modifiers
# ---------------------------------------------------------------------------

_CHAOTIC_INSERTS = [
    _t("WITNESS_1", "[shouting in background] Get away from her! Get him out of here!"),
    _t("OFFICER", "Ma'am ma'am please step back. Sir, sir, sir, drop that."),
    _t("DISPATCH", "Unit 24, additional units requested at your location, "
                  "be advised of multiple parties yelling."),
    _t("COMPLAINANT", "[crying] I can't -- I can't talk right now."),
]

_MULTI_SPEAKER_INSERTS = [
    _t("WITNESS_2", "I was driving past and saw the whole thing through my window."),
    _t("WITNESS_3", "I live across the street. I came out when I heard the noise."),
    _t("WITNESS_2", "It looked like the man in the red shirt threw the first punch.", "incul"),
    _t("WITNESS_3", "Actually from my angle it looked like the other guy started it.", "excul"),
]

_CODE_SWITCHED_INSERTS = [
    _t("COMPLAINANT", "No hablo bien ingles. Estoy muy nerviosa."),
    _t("COMPLAINANT", "He grabbed me, me agarro fuerte por el brazo.", "incul"),
    _t("WITNESS_1", "Necesitamos un interprete por favor."),
]

_EXTENDED_INTERVIEW_INSERTS = [
    _t("OFFICER", "I'm going to step you through the timeline one more time. "
                 "What time did you arrive home today?"),
    _t("SUSPECT", "About two-thirty. I went to the gym first, then came home."),
    _t("OFFICER", "Which gym?"),
    _t("SUSPECT", "Iron Forge on Walnut. I have a receipt for the smoothie I bought after."),
    _t("OFFICER", "Have you had any prior contact with law enforcement?"),
    _t("SUSPECT", "I had a DUI ten years ago. Nothing since."),
    _t("OFFICER", "Is there anything else you want me to include in the report?"),
    _t("SUSPECT", "Yes. I cooperated fully and I never raised my voice.", "excul"),
]


def apply_condition(base: dict, condition: str) -> dict:
    """Return a new transcript dict produced by applying the condition to the base."""
    if condition not in CONDITIONS:
        raise KeyError(f"Unknown condition: {condition}")
    out = deepcopy(base)
    turns = out["turns"]

    if condition == "clean":
        return out
    elif condition == "chaotic":
        # Inject chaotic background turns between the original turns.
        # NOTE: we rebind the local ``turns`` here (not ``out["turns"]``)
        # because the epilogue below does ``out["turns"] = turns``. Writing to
        # ``out["turns"]`` directly would just get overwritten, and the chaotic
        # condition would quietly degenrate into the clean one.
        new = []
        n_inserted = 0
        for i, t in enumerate(turns):
            new.append(t)
            if i in (1, 4, 8):
                # Use each chaotic insert once, in order, so the three
                # injected turns are distinct.
                new.append(deepcopy(_CHAOTIC_INSERTS[n_inserted % len(_CHAOTIC_INSERTS)]))
                n_inserted += 1
        turns = new
    elif condition == "multi_speaker":
        # Append additional bystander witnesses near the middle of the encounter.
        mid = len(turns) // 2
        for j, ins in enumerate(_MULTI_SPEAKER_INSERTS):
            turns.insert(mid + j, deepcopy(ins))
    elif condition == "code_switched":
        # Splice Spanish/English code-switched turns at start and middle.
        for j, ins in enumerate(_CODE_SWITCHED_INSERTS):
            turns.insert(2 + j, deepcopy(ins))
    elif condition == "extended_interview":
        # Append a long interview tail.
        for ins in _EXTENDED_INTERVIEW_INSERTS:
            turns.append(deepcopy(ins))
    out["turns"] = turns
    return out


def all_base_items() -> Iterable[dict]:
    """Yield the 30 (category, condition) base transcripts."""
    for cat in CATEGORIES:
        for cond in CONDITIONS:
            base = base_template(cat)
            item = apply_condition(base, cond)
            item["condition"] = cond
            item["id"] = f"{cat}__{cond}"
            yield item
