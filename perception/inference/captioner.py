"""VLM dense captioning.

Backends (chosen once, at first use — never at import time), controlled by
``CAPTION_BACKEND``:

* ``qwen`` (default): **Qwen2.5-VL-3B-Instruct** — an instruction VLM that in a
  single prompt describes the scene, counts and characterises people, names
  interactions (handshake, handover, etc.), lists objects/vehicles, and reads
  legible text / number plates (OCR). On a capable GPU it loads in 4-bit; on
  CPU it runs in float32 (correct but slow — expect ~15-30 s/frame).
* ``blip``: BLIP-2 (opt-2.7b, 4-bit) on a GPU with >= ``VRAM_THRESHOLD_GB``,
  else BLIP-base (~1 GB, CPU-friendly but weak — generic captions, no actions).
* ``auto``: Qwen when a capable GPU is present, otherwise BLIP-base.

If the preferred backend fails to load (missing deps, OOM), it falls back to
BLIP-base so ingestion never hard-fails on captioning.

Caption deduplication: consecutive keyframes often show the same scene. If the
CLIP cosine similarity between the new caption and the previous one exceeds
``CAPTION_DEDUP_THRESHOLD`` (default 0.92), the previous caption is reused.
"""

from __future__ import annotations

import base64
import io
import json
import logging
import re
from collections import Counter
from typing import Any, Callable

import numpy as np
from PIL import Image

from perception.config import get_device, get_settings, get_vram_gb
from perception.inference.embedder import Embedder
from perception.models import Detection

logger = logging.getLogger(__name__)


# Free-form instruction (BLIP-2 / fallback path).
_QWEN_CAPTION_INSTRUCTION = (
    "You are analysing a single frame from a surveillance video. "
    "Describe it in 1-3 concise sentences, stating concretely:\n"
    "- how many people are visible and what each is doing;\n"
    "- any interaction between people (e.g. shaking hands, handing over an "
    "object, fighting, embracing, talking);\n"
    "- notable objects or vehicles;\n"
    "- any clearly legible text or vehicle number plates (read them out);\n"
    "- any safety hazards (fire, smoke) and whether people are missing PPE "
    "(helmet, hi-vis vest, gloves).\n"
    "Only state what is visibly certain — do not speculate. All output MUST be strictly in English."
)

# Structured instruction — asks the VLM for a strict JSON object we can index
# field-by-field (scene / people / interactions / objects / vehicles / OCR /
# events). Far better for retrieval + summarisation than free-form prose.
_STRUCTURED_INSTRUCTION = (
    "Describe this surveillance frame as ONE JSON object. Use EXACTLY these keys, "
    "no others:\n"
    '{"scene": "<indoor/outdoor, lighting>", '
    '"people": [{"count": <int>, "activity": "<what they do>", '
    '"clothing": ["<colour words>"], "ppe": ["<safety gear worn, empty if none>"]}], '
    '"interactions": ["<person-person or person-object>"], '
    '"objects": ["<carried/notable objects>"], '
    '"vehicles": [{"type": "<car/bike/...>", "color": "<colour>", "plate": "<text or empty>"}], '
    '"text": ["<legible text or plates>"], '
    '"events": ["<fall, fight, running, loitering, intrusion, fire, smoke, missing PPE>"]}\n'
    "Example of a VALID answer (copy this format exactly):\n"
    '{"scene": "indoor parking area, artificial light", '
    '"people": [{"count": 2, "activity": "walking", "clothing": ["blue shirt", "black pants"], '
    '"ppe": ["hi-vis vest"]}], "interactions": [], "objects": ["backpack"], '
    '"vehicles": [], "text": [], "events": []}\n'
    "STRICT RULES: output ONLY the JSON object and nothing else. All text MUST be strictly in English. No markdown (no ** or "
    "backticks or ```), no comments, no trailing commas, no extra keys, and NEVER add "
    "database fields like _id, __v, timeStamp, cameraId, videoUrl, or eventDetails. "
    "Plain colour words, not hex codes. Empty arrays/strings when unsure. Do not invent."
)


# Flat KEY: value instruction — one line per field, no nesting. Small models
# can't misnest an array or forget a bracket here, so it's far more robust than
# JSON on a weak/CPU VLM. Parsed by parse_flat().
_FLAT_INSTRUCTION = (
    "IMPORTANT: Write your entire answer in ENGLISH ONLY. Do not use any Chinese "
    "or non-English characters.\n"
    "Describe this surveillance frame. Output ONLY these lines, one per line, "
    "exactly these keys. Put your answer after the colon; leave it blank if "
    "nothing applies. Do NOT output JSON, brackets, quotes, or markdown.\n"
    "SCENE: <indoor or outdoor, lighting>\n"
    "PEOPLE: <how many people>\n"
    "ACTIVITY: <what the people are doing>\n"
    "CLOTHING: <clothing colours, comma-separated>\n"
    "PPE: <safety gear worn: hi-vis vest, helmet, gloves, mask; blank if none>\n"
    "OBJECTS: <notable or carried objects, comma-separated>\n"
    "INTERACTIONS: <person-person or person-object: e.g. shaking hands, handing "
    "over an object, hugging, fighting, talking together>\n"
    "VEHICLES: <colour and type, comma-separated>\n"
    "PLATES: <number plates, comma-separated>\n"
    "TEXT: <legible signs or labels>\n"
    "EVENTS: <fall, fight, running, loitering, intrusion, fire, smoke, missing PPE>\n"
    "Example:\n"
    "SCENE: indoor parking area, artificial light\n"
    "PEOPLE: 2\n"
    "ACTIVITY: walking\n"
    "CLOTHING: blue shirt, black pants\n"
    "PPE: hi-vis vest\n"
    "OBJECTS: backpack\n"
    "INTERACTIONS: \n"
    "VEHICLES: silver car\n"
    "PLATES: \n"
    "TEXT: \n"
    "EVENTS: \n"
    "Remember: every value must be in English only.\n"
)


# System message pinning English output (Qwen otherwise drifts to Chinese).
_QWEN_SYSTEM_EN = {
    "role": "system",
    "content": "You are a surveillance video analyst. Always respond in English "
               "only. Never use Chinese or any non-English characters.",
}

# Values that mean "nothing" — dropped from parsed flat lists.
_FLAT_EMPTY = {"", "none", "n/a", "na", "nil", "null", "unknown", "-"}


def parse_flat(text: str) -> dict:
    """Parse a flat ``KEY: value`` caption into our stored fields.

    Returns the same shape as :func:`normalize_structured`:
    ``{caption, scene, actions, ocr_text, events, items}``. Bulletproof — there
    is no nesting to break; unknown lines are ignored and it never raises.
    """
    fields: dict[str, str] = {}
    for line in text.splitlines():
        line = line.strip().lstrip("-*• \t").strip()
        if ":" not in line:
            continue
        key, _, val = line.partition(":")
        key = key.strip().strip("*`").strip().lower()
        val = val.strip().strip("*`").strip()
        if key and key not in fields:
            fields[key] = val

    def lst(*keys: str) -> list[str]:
        out: list[str] = []
        for k in keys:
            for part in re.split(r"[;,]", fields.get(k, "")):
                p = part.strip()
                if p and p.lower() not in _FLAT_EMPTY:
                    out.append(p)
        return out

    scene = fields.get("scene", "").strip()
    if scene.lower() in _FLAT_EMPTY:
        scene = ""
    activities = lst("activity", "activities")
    clothing = lst("clothing")
    ppe = lst("ppe")
    objects = lst("objects", "object")
    interactions = lst("interactions", "interaction")
    vehicles = lst("vehicles", "vehicle")
    plates = lst("plates", "plate")
    text_items = lst("text")
    events = lst("events", "event")

    ocr_text = "; ".join(text_items + plates)
    caption_acts = list(activities)
    if ppe:
        caption_acts.append("wearing " + ", ".join(ppe))
    actions = activities + interactions + (["wearing " + ", ".join(ppe)] if ppe else [])
    caption = _synthesize_caption(
        scene, caption_acts, interactions, objects, vehicles, events, ocr_text
    )
    if not caption:
        # Nothing parsed into fields — keep the raw text so it isn't lost.
        caption = text.strip()[:300]
    return {
        "caption": caption,
        "scene": scene,
        "actions": actions,
        "ocr_text": ocr_text,
        "events": events,
        "items": _dedupe_seq(objects + vehicles + ppe + clothing),
    }


# ── Structured-JSON parsing (VLM-agnostic, unit-testable) ──────


def _flatten_item(item: Any) -> str:
    """Render one list item as a readable phrase.

    Strings pass through; dicts (e.g. a VLM returning ``{"type": "car",
    "color": "gray"}`` inside ``objects``) become ``"gray car"`` rather than a
    raw ``{...}`` repr. A ``plate`` field is appended when present.
    """
    if isinstance(item, dict):
        color = str(item.get("color", "")).strip()
        typ = str(item.get("type", "")).strip()
        name = str(item.get("name", "")).strip()
        base = (f"{color} {typ}".strip() or name
                or " ".join(str(v).strip() for v in item.values() if str(v).strip()))
        plate = str(item.get("plate", "")).strip()
        if plate:
            base = f"{base} (plate {plate})".strip()
        return base.strip()
    return str(item).strip()


def _as_str_list(value: Any) -> list[str]:
    """Coerce an arbitrary JSON value into a clean list of non-empty strings."""
    if value is None:
        return []
    if isinstance(value, list):
        return [s for s in (_flatten_item(x) for x in value) if s]
    text = str(value).strip()
    return [text] if text else []


def _dedupe_seq(items: list[str]) -> list[str]:
    """Case-insensitive dedupe, first-seen order. VLMs re-list the same car
    many times ("white car" ×25) — collapse to the distinct set."""
    seen: set[str] = set()
    out: list[str] = []
    for x in items:
        k = x.lower()
        if k not in seen:
            seen.add(k)
            out.append(x)
    return out


def extract_json(text: str) -> dict:
    """Best-effort extraction of a single JSON object from VLM output.

    Scans from the first ``{`` tracking string/bracket state so it ignores
    markdown fences and surrounding prose. Crucially, it **repairs truncated**
    output (a common failure when the token budget runs out mid-JSON) by
    closing an unterminated string, dropping a dangling comma, and appending
    the missing closing brackets. Returns ``{}`` only when nothing salvages.
    """
    if not text:
        return {}
    start = text.find("{")
    if start == -1:
        return {}
    body = text[start:]

    stack: list[str] = []
    in_str = False
    esc = False
    end: int | None = None
    for i, ch in enumerate(body):
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch in "{[":
            stack.append("}" if ch == "{" else "]")
        elif ch in "}]":
            if stack:
                stack.pop()
            if not stack:
                end = i + 1
                break

    if end is not None:
        candidate = body[:end]
    else:
        # Truncated output: close the open string, drop any trailing comma,
        # then append the outstanding closers (innermost first).
        candidate = body + ('"' if in_str else "")
        candidate = candidate.rstrip().rstrip(",").rstrip()
        # Cut off right after a key ("color":) with no value → drop the dangling
        # key back to the previous separator so the JSON stays valid.
        if candidate.endswith(":"):
            sep = max(candidate.rfind(","), candidate.rfind("{"), candidate.rfind("["))
            candidate = candidate[: sep + 1] if sep != -1 else candidate
        candidate = candidate.rstrip().rstrip(",")
        candidate += "".join(reversed(stack))

    # Models sometimes add // or /* */ comments and trailing commas — strip them
    # so the JSON (or Python-dict) parses. (Hex strings like '#4a6e79' keep their
    # '#' since it's inside quotes; we only remove // and /* */ styles.)
    import re

    candidate = candidate.replace("**", "").replace("`", "")   # markdown bold / code
    candidate = re.sub(r"/\*.*?\*/", "", candidate, flags=re.S)
    candidate = re.sub(r"//[^\n]*", "", candidate)
    candidate = re.sub(r",(\s*[}\]])", r"\1", candidate)

    try:
        parsed = json.loads(candidate)
    except json.JSONDecodeError:
        # Some VLMs emit Python-style dicts with single quotes (invalid JSON).
        # ast.literal_eval parses those safely (no code execution).
        try:
            import ast

            parsed = ast.literal_eval(candidate)
        except (ValueError, SyntaxError):
            return {}
    return parsed if isinstance(parsed, dict) else {}


def _synthesize_caption(
    scene: str,
    activities: list[str],
    interactions: list[str],
    objects: list[str],
    vehicles: list[str],
    events: list[str],
    ocr_text: str,
) -> str:
    """Flatten structured fields into one readable sentence for embed + display."""
    parts: list[str] = []
    if scene:
        parts.append(scene[0].upper() + scene[1:] + ".")
    if activities:
        parts.append("People: " + "; ".join(activities) + ".")
    if interactions:
        parts.append("Interactions: " + "; ".join(interactions) + ".")
    if objects:
        parts.append("Objects: " + ", ".join(objects) + ".")
    if vehicles:
        parts.append("Vehicles: " + "; ".join(vehicles) + ".")
    if events:
        parts.append("Events: " + ", ".join(events) + ".")
    if ocr_text:
        parts.append("Text: " + ocr_text + ".")
    return " ".join(parts).strip()


def normalize_structured(data: dict) -> dict:
    """Turn a parsed VLM JSON object into our stored fields.

    Returns keys: ``caption`` (synthesized NL), ``scene``, ``actions``
    (activities + interactions), ``ocr_text``, ``events``. Defensive against
    the many shapes VLMs emit (dicts vs strings, singular vs plural keys).
    """
    scene = data.get("scene", "")
    if isinstance(scene, dict):
        # Join the values only ("indoor, natural"), not the keys.
        scene = ", ".join(str(v).strip() for v in scene.values() if str(v).strip())
    scene = str(scene).strip()

    activities: list[str] = []
    ppe_items: list[str] = []
    people = data.get("people", [])
    if isinstance(people, list):
        for person in people:
            if isinstance(person, dict):
                act = person.get("activity") or person.get("activities")
                if act:
                    activities.append(str(act).strip())
                ppe_items.extend(_as_str_list(person.get("ppe")))
            elif str(person).strip():
                activities.append(str(person).strip())
    ppe_items = _dedupe_seq(ppe_items)
    if ppe_items:
        # Fold worn PPE into the description so it's searchable + displayed
        # (e.g. "wearing hi-vis vest") and answerable in Q&A.
        activities.append("wearing " + ", ".join(ppe_items))

    interactions = _as_str_list(data.get("interactions") or data.get("interaction"))
    objects = _as_str_list(data.get("objects"))
    events = _as_str_list(data.get("events"))
    text_items = _as_str_list(data.get("text"))

    vehicle_strs: list[str] = []   # with plate, for the caption text
    vehicle_names: list[str] = []  # colour+type only, for the items list
    plates: list[str] = []
    vehicles = data.get("vehicles", [])
    if isinstance(vehicles, list):
        for veh in vehicles:
            if isinstance(veh, dict):
                name = " ".join(
                    p for p in (str(veh.get("color", "")).strip(),
                                str(veh.get("type", "")).strip()) if p
                )
                desc = name
                plate = str(veh.get("plate", "")).strip()
                if plate:
                    plates.append(plate)
                    desc = f"{name} (plate {plate})".strip()
                if desc:
                    vehicle_strs.append(desc)
                if name:
                    vehicle_names.append(name)
            elif str(veh).strip():
                vehicle_strs.append(str(veh).strip())
                vehicle_names.append(str(veh).strip())

    # Drop objects already covered by the vehicles list (VLMs often duplicate
    # vehicles into both), so the caption doesn't say them twice.
    veh_lower = {v.lower() for v in vehicle_strs}
    objects = _dedupe_seq([o for o in objects if o.lower() not in veh_lower])
    # VLMs re-list the same car dozens of times — keep the distinct set so the
    # caption and Items reflect the ~5 real cars, not 35 duplicates.
    vehicle_strs = _dedupe_seq(vehicle_strs)
    vehicle_names = _dedupe_seq(vehicle_names)

    ocr_text = "; ".join(text_items + plates)
    actions = activities + interactions
    caption = _synthesize_caption(
        scene, activities, interactions, objects, vehicle_strs, events, ocr_text
    )
    return {
        "caption": caption,
        "scene": scene,
        "actions": actions,
        "ocr_text": ocr_text,
        "events": events,
        # Per-object colour phrases from the VLM (e.g. "orange backpack",
        # "silver car") — the accurate source for the UI "Items" list.
        "items": _dedupe_seq(objects + vehicle_names + ppe_items),
    }


def _empty_description(caption: str = "") -> dict:
    return {
        "caption": caption, "scene": "", "actions": [],
        "ocr_text": "", "events": [], "items": [],
    }


class Captioner:
    """Lazy VLM wrapper producing one dense caption string per frame."""

    def __init__(self, embedder: Embedder | None = None) -> None:
        settings = get_settings()
        self.backend_pref = settings.caption_backend.lower()
        self.caption_format = settings.caption_format.lower()
        self.caption_device = settings.caption_device.lower()
        self.qwen_name = settings.qwen_vl_model
        self.max_new_tokens = settings.caption_max_new_tokens
        self.qwen_min_pixels = settings.qwen_min_pixels
        self.qwen_max_pixels = settings.qwen_max_pixels
        self.blip2_name = settings.blip2_model
        self.blip_base_name = settings.blip_base_model
        self.vram_threshold_gb = settings.vram_threshold_gb
        self.dedup_threshold = settings.caption_dedup_threshold
        # Remote Ollama vision server (CAPTION_BACKEND=ollama). Falls back to the
        # main Ollama URL when no dedicated caption server is configured.
        caption_url = settings.ollama_caption_base_url or settings.ollama_base_url
        self.ollama_caption_url = caption_url.rstrip("/")
        self.ollama_caption_model = settings.ollama_caption_model
        self._concurrency = max(1, settings.ollama_concurrency)
        self._batch_size = max(1, settings.caption_batch_size)
        # Shared with the pipeline's embedder so CLIP loads only once.
        self._embedder = embedder or Embedder()
        self._model: Any = None
        self._processor: Any = None
        self._device = "cpu"
        # "qwen" | "blip2" | "blip" | "ollama"
        self._backend = ""

    # ── Model loading ──────────────────────────────────────────

    def _load(self) -> None:
        # Ollama is a remote HTTP service — nothing to load locally.
        if self._backend == "ollama" or self._model is not None:
            return
        if self.backend_pref == "ollama":
            logger.info(
                "Caption backend=ollama: using %s at %s",
                self.ollama_caption_model, self.ollama_caption_url,
            )
            self._backend = "ollama"
            return
        device = get_device()
        vram = get_vram_gb()
        # CAPTION_DEVICE overrides the auto GPU/CPU choice. "cpu" forces the VLM
        # onto CPU (reliable on a tiny/shared GPU that can't hold a 3B VLM); the
        # small GPU stays free for CLIP/YOLO.
        if self.caption_device == "cpu":
            device = "cpu"
        capable_gpu = device == "cuda" and (
            self.caption_device == "cuda" or vram >= self.vram_threshold_gb
        )

        if self.backend_pref in ("qwen", "auto"):
            if self.backend_pref == "auto" and not capable_gpu:
                logger.info("caption_backend=auto on CPU/small GPU: using BLIP base")
                self._load_blip_base(device)
                return
            try:
                self._load_qwen(device, capable_gpu)
                return
            except Exception:
                logger.exception("Qwen2.5-VL load failed; falling back to BLIP")

        # Explicit "blip" backend, or a Qwen fallback.
        if capable_gpu:
            try:
                self._load_blip2()
                return
            except Exception:
                logger.exception("BLIP-2 load failed; falling back to BLIP base")
        else:
            logger.info(
                "VRAM %.1f GB < %.1f GB threshold (or CPU-only): using BLIP base",
                vram, self.vram_threshold_gb,
            )
        self._load_blip_base(device)

    def _load_qwen(self, device: str, capable_gpu: bool) -> None:
        import torch
        from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

        logger.info("Loading Qwen2.5-VL (%s)", self.qwen_name)
        self._processor = AutoProcessor.from_pretrained(
            self.qwen_name,
            min_pixels=self.qwen_min_pixels,
            max_pixels=self.qwen_max_pixels,
        )

        load_kwargs: dict[str, Any] = {}
        if capable_gpu:
            from transformers import BitsAndBytesConfig

            load_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_quant_type="nf4",
                # Double quantization: also quantizes the quant constants,
                # saving ~0.4 GB — the margin that lets Qwen-3B load on a 4 GB card.
                bnb_4bit_use_double_quant=True,
            )
            # Use {"":"cuda:0"} instead of "auto".
            # device_map="auto" makes accelerate reserve only 90% of VRAM and
            # offload the remainder to CPU, but bitsandbytes 4-bit quantization
            # cannot run on CPU-dispatched layers → raises ValueError.
            # {"":"cuda:0"} forces ALL layers onto GPU 0 directly, bypassing
            # the auto-split. Qwen2.5-VL-3B in 4-bit is ~2.5 GB — fits in the
            # GTX 1650's 3.62 GB budget without any CPU offloading.
            load_kwargs["device_map"] = {"": "cuda:0"}
            self._device = "cuda"
        elif device == "cuda":
            load_kwargs["dtype"] = torch.float16
            self._device = "cuda"
        else:
            load_kwargs["dtype"] = torch.float32
            self._device = "cpu"
            logger.warning(
                "Qwen2.5-VL on CPU is slow (~15-30 s/frame). Consider a GPU, a "
                "smaller MAX_KEYFRAMES, or CAPTION_BACKEND=blip for faster ingest."
            )

        # Left-padding is required for correct *batched* decoder generation:
        # padded rows share one length, so trimming the prompt is uniform and
        # generated tokens are appended after the real prompt (not after pad).
        tokenizer = getattr(self._processor, "tokenizer", None)
        if tokenizer is not None:
            tokenizer.padding_side = "left"

        self._model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            self.qwen_name, **load_kwargs
        )
        if "device_map" not in load_kwargs:
            self._model = self._model.to(self._device)
        self._model.eval()
        self._backend = "qwen"

    def _load_blip2(self) -> None:
        import torch
        from transformers import Blip2ForConditionalGeneration, Blip2Processor, BitsAndBytesConfig

        logger.info("Loading BLIP-2 (%s) in 4-bit", self.blip2_name)
        quant = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type="nf4",
        )
        self._processor = Blip2Processor.from_pretrained(self.blip2_name)
        self._model = Blip2ForConditionalGeneration.from_pretrained(
            self.blip2_name, quantization_config=quant, device_map="auto"
        )
        self._device = "cuda"
        self._backend = "blip2"

    def _load_blip_base(self, device: str) -> None:
        from transformers import BlipForConditionalGeneration, BlipProcessor

        logger.info("Loading BLIP base (%s) on %s", self.blip_base_name, device)
        self._processor = BlipProcessor.from_pretrained(self.blip_base_name)
        self._model = BlipForConditionalGeneration.from_pretrained(self.blip_base_name)
        self._model = self._model.to(device).eval()
        self._device = device
        self._backend = "blip"

    # ── Captioning ─────────────────────────────────────────────

    @staticmethod
    def _build_hint(detections: list[Detection]) -> str:
        """Format detections into a readable label string, e.g. "2 persons, 1 car"."""
        if not detections:
            return ""
        counts = Counter(d.label for d in detections)
        parts = []
        for label in sorted(counts):
            n = counts[label]
            plural = label if n == 1 or label.endswith("s") else label + "s"
            parts.append(f"{n} {plural}")
        return ", ".join(parts)

    def _qwen_generate(self, image: Image.Image, prompt: str) -> str:
        """Run one Qwen2.5-VL generation for an arbitrary text prompt + image."""
        import torch
        from qwen_vl_utils import process_vision_info

        messages = [
            _QWEN_SYSTEM_EN,
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image.convert("RGB")},
                    {"type": "text", "text": prompt},
                ],
            },
        ]
        text = self._processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = self._processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        ).to(self._device)

        # Free any cached/fragmented VRAM before the 4-bit matmul so bitsandbytes
        # can allocate the dequantised weight buffer without hitting an OOM.
        if self._device == "cuda":
            torch.cuda.empty_cache()

        with torch.inference_mode():
            generated = self._model.generate(
                **inputs, max_new_tokens=self.max_new_tokens, do_sample=False,
                repetition_penalty=1.15,
                # no_repeat_ngram_size intentionally omitted: it uses CUDA-side
                # fancy indexing that fails after a prior OOM leaves the device
                # context in a bad state. repetition_penalty handles repetition.
            )
        # Detach the generated ids from the computation graph before freeing inputs.
        generated = generated.detach().cpu()
        # Release the input tensors from VRAM immediately — they are large
        # (vision tokens) and no longer needed once generation is done.
        input_ids = inputs.input_ids.cpu()
        del inputs
        if self._device == "cuda":
            torch.cuda.empty_cache()

        trimmed = [out[len(inp):] for inp, out in zip(input_ids, generated)]
        text_out = self._processor.batch_decode(
            trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0]
        return " ".join(text_out.split()).strip()

    def _qwen_generate_batch(
        self, images: list[Image.Image], prompts: list[str]
    ) -> list[str]:
        """Generate for a batch of (image, prompt) pairs in one forward pass.

        With left-padding (set at load time) all rows share the padded prompt
        length, so ``generated[:, prompt_len:]`` isolates each row's new tokens.
        """
        import torch
        from qwen_vl_utils import process_vision_info

        conversations = [
            [
                _QWEN_SYSTEM_EN,
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": img.convert("RGB")},
                        {"type": "text", "text": prompt},
                    ],
                },
            ]
            for img, prompt in zip(images, prompts)
        ]
        texts = [
            self._processor.apply_chat_template(c, tokenize=False, add_generation_prompt=True)
            for c in conversations
        ]
        image_inputs, video_inputs = process_vision_info(conversations)
        inputs = self._processor(
            text=texts,
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        ).to(self._device)

        # Free cached VRAM before each batched generation pass.
        if self._device == "cuda":
            torch.cuda.empty_cache()

        with torch.inference_mode():
            generated = self._model.generate(
                **inputs, max_new_tokens=self.max_new_tokens, do_sample=False,
                repetition_penalty=1.15,
            )
        prompt_len = inputs.input_ids.shape[1]
        trimmed = generated[:, prompt_len:]
        decoded = self._processor.batch_decode(
            trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        return [" ".join(t.split()).strip() for t in decoded]

    def _ollama_generate(self, image: Image.Image, prompt: str, force_json: bool = False) -> str:
        """Run one caption against the remote Ollama vision model."""
        import httpx

        buf = io.BytesIO()
        image.convert("RGB").save(buf, format="JPEG", quality=85)
        b64 = base64.b64encode(buf.getvalue()).decode()

        payload: dict[str, Any] = {
            "model": self.ollama_caption_model,
            "messages": [{"role": "user", "content": prompt, "images": [b64]}],
            "stream": False,
            "options": {"num_predict": self.max_new_tokens, "repeat_penalty": 1.15},
        }
        if force_json:
            # Ask Ollama to constrain the decoder to valid JSON.
            payload["format"] = "json"

        resp = httpx.post(f"{self.ollama_caption_url}/api/chat", json=payload, timeout=180)
        if resp.status_code != 200:
            raise RuntimeError(
                f"Ollama caption request to {self.ollama_caption_url} for model "
                f"'{self.ollama_caption_model}' failed ({resp.status_code}): {resp.text}"
            )
        content = resp.json()["message"]["content"]
        return " ".join(content.split()).strip()

    def _caption_qwen(self, image: Image.Image, hint: str) -> str:
        prompt = _QWEN_CAPTION_INSTRUCTION
        if hint:
            prompt += f"\n\nAn object detector reported: {hint}."
        return self._qwen_generate(image, prompt)

    def _caption_blip(self, image: Image.Image, hint: str = "") -> str:
        import torch

        rgb = image.convert("RGB")
        if self._backend == "blip2":
            text = (
                "Question: Describe this scene in detail, including what the people are doing. "
                f"Detected objects: {hint}. Answer:"
                if hint
                else "Question: Describe this scene in detail, including what the people are doing. Answer:"
            )
            inputs = self._processor(images=rgb, text=text, return_tensors="pt")
        else:
            text = "In this security camera scene," if hint else ""
            inputs = (
                self._processor(images=rgb, text=text, return_tensors="pt")
                if text
                else self._processor(images=rgb, return_tensors="pt")
            )

        inputs = {k: v.to(self._device) for k, v in inputs.items()}
        with torch.no_grad():
            output_ids = self._model.generate(**inputs, max_new_tokens=60)
        caption = self._processor.batch_decode(output_ids, skip_special_tokens=True)[0]
        if self._backend == "blip2" and "Answer:" in caption:
            caption = caption.split("Answer:", 1)[-1]
        return caption.strip()

    def _caption_one(self, image: Image.Image, hint: str = "") -> str:
        if self._backend == "qwen":
            return self._caption_qwen(image, hint)
        if self._backend == "ollama":
            prompt = _QWEN_CAPTION_INSTRUCTION
            if hint:
                prompt += f"\n\nAn object detector reported: {hint}."
            return self._ollama_generate(image, prompt)
        return self._caption_blip(image, hint)

    # ── Structured description (VLM backends) ──────────────────

    @property
    def _instruction(self) -> str:
        """The VLM prompt for the active caption format."""
        return _FLAT_INSTRUCTION if self.caption_format == "flat" else _STRUCTURED_INSTRUCTION

    def _parse_raw(self, raw: str) -> dict:
        """Parse one raw VLM response into our stored fields, per caption_format."""
        if self.caption_format == "flat":
            return parse_flat(raw)  # bulletproof: never fails, no raw-blob fallback
        data = extract_json(raw)
        if not data:
            logger.warning("Structured caption JSON parse failed; using raw text")
            return _empty_description(raw)
        return normalize_structured(data)

    def _describe_one(self, image: Image.Image, hint: str) -> dict:
        """Return structured fields for one frame.

        VLM backends (qwen/ollama) are asked for the active caption format and
        parsed by :meth:`_parse_raw`. BLIP can't follow a schema, so it yields a
        free-form caption with empty structured fields.
        """
        if self._backend not in ("qwen", "ollama"):
            return _empty_description(self._caption_blip(image, hint))

        prompt = self._instruction
        if hint:
            prompt += f"\n\nAn object detector reported: {hint}."

        if self._backend == "ollama":
            # Only constrain Ollama to JSON when we actually asked for JSON.
            raw = self._ollama_generate(
                image, prompt, force_json=(self.caption_format == "json")
            )
        else:
            raw = self._qwen_generate(image, prompt)

        return self._parse_raw(raw)

    def _safe_describe_one(self, image: Image.Image, hint: str) -> dict:
        try:
            return self._describe_one(image, hint)
        except Exception as exc:
            logger.exception("Description failed on a frame; using empty")
            # After any CUDA error (OOM, unknown error, etc.) the device context
            # may be corrupted. Synchronise + flush so subsequent frames get a
            # clean CUDA state instead of cascading failures.
            if self._device == "cuda" and "cuda" in str(exc).lower():
                try:
                    import torch
                    torch.cuda.synchronize()
                except Exception:
                    pass
                try:
                    import torch
                    torch.cuda.empty_cache()
                except Exception:
                    pass
            return _empty_description()

    def _fields_from_raw(self, raw: str) -> dict:
        """Parse one raw VLM response into structured fields (format-aware)."""
        return self._parse_raw(raw)

    def _describe_qwen_batched(
        self,
        images: list[Image.Image],
        hints: list[str],
        batch_size: int,
        progress_cb: Callable[[int, int], None] | None = None,
    ) -> list[dict]:
        """Describe frames in Qwen batches (one forward pass per group).

        Trades RAM for speed: a bigger batch keeps the CPU/GPU busier per call
        but raises peak memory. A failed batch falls back to per-frame so one
        bad group never loses the whole run.
        """
        prompts = [
            self._instruction + (f"\n\nAn object detector reported: {h}." if h else "")
            for h in hints
        ]
        total = len(images)
        results: list[dict] = []
        done = 0
        for start in range(0, total, batch_size):
            end = min(start + batch_size, total)
            logger.info("Captioning frames %d-%d/%d (batch of %d)", start + 1, end, total, end - start)
            try:
                raws = self._qwen_generate_batch(images[start:end], prompts[start:end])
                results.extend(self._fields_from_raw(r) for r in raws)
                done += end - start
                if progress_cb is not None:
                    progress_cb(done, total)
            except Exception:
                logger.exception("Batch %d-%d failed; retrying that batch per-frame", start + 1, end)
                for img, hint in zip(images[start:end], hints[start:end]):
                    results.append(self._safe_describe_one(img, hint))
                    done += 1
                    if progress_cb is not None:
                        progress_cb(done, total)
        return results

    def describe_frames(
        self,
        images: list[Image.Image],
        detections: list[list[Detection]] | None = None,
        context: list[str] | None = None,
        progress_cb: Callable[[int, int], None] | None = None,
    ) -> list[dict]:
        """Structured per-frame descriptions (scene/actions/ocr_text/events + caption).

        Stage 1 produces one description per frame. For the network (Ollama)
        backend these run concurrently (they are I/O-bound HTTP calls, so the
        GIL is released while waiting); local CPU/GPU models run sequentially,
        since one inference already uses all cores. Stage 2 deduplicates
        near-identical consecutive frames by caption CLIP cosine, reusing the
        previous description to avoid embedding-store noise.
        """
        self._load()
        # Per-frame grounding hint = object counts + (optional) track facts,
        # e.g. "2 persons, 1 car. Tracked: person_1 (present since 3s)".
        # Track facts are precomputed, so they work in every path below.
        ctx = context or [""] * len(images)
        hints = []
        for i in range(len(images)):
            h = self._build_hint(detections[i]) if detections else ""
            if ctx[i]:
                h = f"{h}. Tracked: {ctx[i]}" if h else f"Tracked: {ctx[i]}"
            hints.append(h)

        # ── Stage 1: per-frame description ──────────────────────
        # - Ollama (network): concurrent HTTP requests.
        # - Local Qwen with CAPTION_BATCH_SIZE > 1: batched forward passes.
        # - Otherwise: sequential (one inference already uses all cores).
        total = len(images)
        if self._backend == "ollama" and self._concurrency > 1 and total > 1:
            from concurrent.futures import ThreadPoolExecutor, as_completed

            raw: list[dict] = [_empty_description()] * total
            with ThreadPoolExecutor(max_workers=self._concurrency) as pool:
                futures = {
                    pool.submit(self._safe_describe_one, img, h): i
                    for i, (img, h) in enumerate(zip(images, hints))
                }
                done = 0
                for fut in as_completed(futures):
                    i = futures[fut]
                    raw[i] = fut.result()
                    done += 1
                    if progress_cb is not None:
                        progress_cb(done, total)
                    logger.info("Captioned %d/%d frames (%d-way concurrent)", done, total, self._concurrency)
        elif self._backend == "qwen" and self._batch_size > 1 and total > 1:
            raw = self._describe_qwen_batched(
                images, hints, self._batch_size, progress_cb=progress_cb
            )
        else:
            # Sequential path can also thread the PREVIOUS caption for
            # continuity (batched/concurrent paths can't — the prior result
            # isn't ready yet, so they use track facts only).
            raw = []
            prev_cap = ""
            for i, (img, h) in enumerate(zip(images, hints)):
                logger.info("Captioning frame %d/%d (%d remaining)", i + 1, total, total - i - 1)
                h2 = f"{h}. Previous keyframe showed: {prev_cap}" if prev_cap else h
                fields = self._safe_describe_one(img, h2)
                raw.append(fields)
                prev_cap = (fields.get("caption", "") or "")[:200]
                if progress_cb is not None:
                    progress_cb(i + 1, total)

        # ── Stage 2: sequential dedup ───────────────────────────
        results: list[dict] = []
        prev: dict | None = None
        prev_vec: np.ndarray | None = None
        for fields in raw:
            caption = fields.get("caption", "").strip()
            if caption and prev_vec is not None and prev is not None:
                try:
                    vec = self._embedder.embed_text(caption)
                    if float(vec @ prev_vec) > self.dedup_threshold:
                        results.append(prev)  # reuse previous description verbatim
                        continue
                    prev_vec = vec
                except Exception:
                    prev_vec = None
            elif caption:
                try:
                    prev_vec = self._embedder.embed_text(caption)
                except Exception:
                    prev_vec = None

            prev = fields
            results.append(fields)

        return results

    def caption_frames(
        self,
        images: list[Image.Image],
        detections: list[list[Detection]] | None = None,
    ) -> list[str]:
        """Caption every frame, deduplicating near-identical consecutive captions.

        When ``detections`` are supplied, detected object labels/counts are fed
        to the VLM as grounding so captions reference what was actually found.
        Failures on individual frames are logged and yield an empty caption
        rather than aborting the batch.
        """
        self._load()
        captions: list[str] = []
        prev_caption = ""
        prev_vec: np.ndarray | None = None

        for i, image in enumerate(images):
            hint = self._build_hint(detections[i]) if detections else ""
            try:
                caption = self._caption_one(image, hint=hint)
            except Exception:
                logger.exception("Captioning failed on frame %d; using empty caption", i)
                captions.append("")
                continue

            if caption and prev_vec is not None:
                try:
                    vec = self._embedder.embed_text(caption)
                    similarity = float(vec @ prev_vec)
                    if similarity > self.dedup_threshold:
                        logger.debug(
                            "Frame %d caption ~ previous (cos=%.3f); reusing '%s'",
                            i, similarity, prev_caption,
                        )
                        captions.append(prev_caption)
                        continue
                    prev_vec = vec
                except Exception:
                    logger.exception("Caption dedup embedding failed; keeping caption")
                    prev_vec = None
            elif caption:
                try:
                    prev_vec = self._embedder.embed_text(caption)
                except Exception:
                    prev_vec = None

            prev_caption = caption
            captions.append(caption)

        return captions

    def caption(self, image: Image.Image) -> str:
        """Convenience wrapper for a single image."""
        return self.caption_frames([image])[0]

    def caption_actions(
        self,
        images: list[Image.Image],
        detections: list[list[Detection]] | None = None,
    ) -> list[list[str]]:
        """Return per-frame action descriptions.

        Qwen2.5-VL already folds actions/interactions into its main caption, so
        this returns empties for the Qwen backend (no slow second pass). For the
        BLIP backends it runs a focused VQA-style pass on frames with 2+ people.
        """
        self._load()

        if self._backend in ("qwen", "ollama"):
            # Actions are captured in the primary structured caption already.
            return [[] for _ in images]

        results: list[list[str]] = []
        for i, image in enumerate(images):
            person_count = 0
            if detections:
                person_count = sum(
                    1 for d in detections[i] if d.label.lower() == "person"
                )
            if person_count < 2 and detections is not None:
                results.append([])
                continue
            try:
                action = self._caption_action_one(image)
                results.append([action] if action else [])
            except Exception:
                logger.exception("Action captioning failed on frame %d", i)
                results.append([])
        return results

    def _caption_action_one(self, image: Image.Image) -> str:
        """Run one VQA-style caption focused on what people are doing (BLIP)."""
        import torch

        rgb = image.convert("RGB")
        if self._backend == "blip2":
            text = (
                "Question: What are the people doing? "
                "Are they shaking hands, fighting, embracing, or interacting in any way? "
                "Answer:"
            )
            inputs = self._processor(images=rgb, text=text, return_tensors="pt")
        else:
            text = "The people in this scene are"
            inputs = self._processor(images=rgb, text=text, return_tensors="pt")

        inputs = {k: v.to(self._device) for k, v in inputs.items()}
        with torch.no_grad():
            output_ids = self._model.generate(**inputs, max_new_tokens=40)
        answer = self._processor.batch_decode(output_ids, skip_special_tokens=True)[0]
        if self._backend == "blip2" and "Answer:" in answer:
            answer = answer.split("Answer:", 1)[-1]
        return answer.strip()


if __name__ == "__main__":
    import argparse

    from perception.config import configure_logging

    configure_logging()
    parser = argparse.ArgumentParser(description="Caption a single image with the active VLM")
    parser.add_argument("image", help="path to an image file")
    args = parser.parse_args()

    captioner = Captioner()
    print(captioner.caption(Image.open(args.image).convert("RGB")))
