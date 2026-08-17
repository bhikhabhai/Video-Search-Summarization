"""Unit tests for structured-caption JSON parsing (no models required)."""

from __future__ import annotations

from perception.inference.captioner import extract_json, normalize_structured


class TestExtractJson:
    def test_plain_object(self):
        assert extract_json('{"scene": "parking"}') == {"scene": "parking"}

    def test_markdown_fenced(self):
        raw = '```json\n{"scene": "lobby", "events": ["fall"]}\n```'
        assert extract_json(raw) == {"scene": "lobby", "events": ["fall"]}

    def test_prose_around_json(self):
        raw = 'Here is the result: {"objects": ["bag"]} — hope that helps!'
        assert extract_json(raw) == {"objects": ["bag"]}

    def test_garbage_returns_empty(self):
        assert extract_json("not json at all") == {}
        assert extract_json("") == {}

    def test_non_object_returns_empty(self):
        # A JSON array is not a dict object.
        assert extract_json("[1, 2, 3]") == {}

    def test_truncated_object_is_repaired(self):
        # Token budget ran out mid-object: unterminated string + open brackets.
        raw = '{"scene": "parking area", "objects": ["bag", "cone'
        out = extract_json(raw)
        assert out["scene"] == "parking area"
        assert out["objects"][0] == "bag"

    def test_truncated_with_trailing_comma(self):
        raw = '{"scene": "lobby", "events": ["fall"],'
        out = extract_json(raw)
        assert out["scene"] == "lobby"
        assert out["events"] == ["fall"]

    def test_truncated_nested_arrays(self):
        raw = '{"people": [{"count": 2, "activity": "standing'
        out = extract_json(raw)
        assert out["people"][0]["count"] == 2

    def test_truncated_after_dangling_key(self):
        # Cut off right after a key with no value (real Qwen failure).
        raw = ('{"objects": [{"type": "backpack", "color": "orange"}], '
               '"vehicles": [{"type": "car", "color": "silver"}, '
               '{"type": "car", "color":')
        out = extract_json(raw)
        assert out["objects"][0]["color"] == "orange"
        assert out["vehicles"][0]["type"] == "car"  # complete vehicle kept

    def test_trailing_prose_after_object(self):
        # Balanced object followed by junk should still parse cleanly.
        raw = '{"scene": "street"} Note: this is my best guess.'
        assert extract_json(raw) == {"scene": "street"}


class TestNormalizeStructured:
    def test_full_object(self):
        data = {
            "scene": "parking area, outdoor, daytime",
            "people": [{"count": 2, "activity": "standing near vehicle"}],
            "interactions": ["one person handing a bag to another"],
            "objects": ["bag"],
            "vehicles": [{"type": "car", "color": "white", "plate": "GJ01AB1234"}],
            "text": ["EXIT"],
            "events": ["object handover"],
        }
        out = normalize_structured(data)
        assert out["scene"].startswith("parking area")
        # activities + interactions merged into actions
        assert "standing near vehicle" in out["actions"]
        assert "one person handing a bag to another" in out["actions"]
        # OCR text combines the text[] items and vehicle plates
        assert "EXIT" in out["ocr_text"]
        assert "GJ01AB1234" in out["ocr_text"]
        assert out["events"] == ["object handover"]
        # Synthesized caption should be a non-empty readable string mentioning key facts
        assert "white car" in out["caption"]
        assert "GJ01AB1234" in out["caption"]
        assert "object handover" in out["caption"]

    def test_empty_object(self):
        out = normalize_structured({})
        assert out == {
            "caption": "",
            "scene": "",
            "actions": [],
            "ocr_text": "",
            "events": [],
            "items": [],
        }

    def test_items_carry_vlm_colours(self):
        out = normalize_structured({
            "objects": [{"type": "backpack", "color": "orange"}],
            "vehicles": [{"type": "car", "color": "silver", "plate": "AB12"}],
        })
        assert "orange backpack" in out["items"]
        assert "silver car" in out["items"]        # no plate noise in items

    def test_repeated_vehicles_deduped(self):
        # VLM re-lists the same cars many times → distinct set only.
        out = normalize_structured({
            "vehicles": [{"type": "car", "color": "white"}] * 25
                        + [{"type": "car", "color": "blue"}] * 5,
        })
        assert out["items"] == ["white car", "blue car"]
        assert out["caption"].count("white car") == 1


    def test_scene_as_dict_joins_values_only(self):
        out = normalize_structured({"scene": {"location": "street", "lighting": "night"}})
        # Values only — no "location:" / "lighting:" key noise.
        assert out["scene"] == "street, night"
        assert "location" not in out["scene"]

    def test_objects_as_dicts_are_flattened(self):
        # A VLM sometimes puts vehicle dicts inside `objects` — must not render
        # as raw {'type': ...} reprs.
        out = normalize_structured({
            "objects": [
                {"type": "car", "color": "gray", "plate": ""},
                {"type": "truck", "color": "black", "plate": ""},
            ],
        })
        assert "{" not in out["caption"]
        assert "gray car" in out["caption"]
        assert "black truck" in out["caption"]

    def test_objects_deduped_against_vehicles(self):
        out = normalize_structured({
            "objects": [{"type": "car", "color": "white"}],
            "vehicles": [{"type": "car", "color": "white"}],
        })
        # "white car" should appear once (as a vehicle), not in both sections.
        assert out["caption"].count("white car") == 1

    def test_singular_interaction_key(self):
        out = normalize_structured({"interaction": "handshake"})
        assert "handshake" in out["actions"]

    def test_string_valued_lists(self):
        # A VLM sometimes returns a bare string where a list is expected.
        out = normalize_structured({"objects": "backpack", "events": "running"})
        assert out["events"] == ["running"]
        assert "backpack" in out["caption"]

    def test_vehicle_without_plate(self):
        out = normalize_structured({"vehicles": [{"type": "bike", "color": "red"}]})
        assert "red bike" in out["caption"]
        assert out["ocr_text"] == ""  # no plate, no text
