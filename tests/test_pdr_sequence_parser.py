from solutions.PDR.net import Petri
from solutions.PDR.parse_sequences import build_single_replay_payload


def test_build_single_replay_payload_inserts_wait_5s_between_large_gaps() -> None:
    records = [
        {"transition": "u_LP_TM2", "fire_time": 5},
        {"transition": "t_TM2_PM7", "fire_time": 18},
    ]

    payload = build_single_replay_payload(records)
    sequence = payload["sequence"]

    assert payload["schema_version"] == 2
    assert payload["device_mode"] == "single"
    assert [item["action"] for item in sequence] == [
        "u_LP_TM2",
        "WAIT_5s",
        "WAIT_5s",
        "t_TM2_PM7",
    ]
    assert [item["time"] for item in sequence] == [5, 10, 15, 18]


def test_petri_search_records_transition_and_fire_time() -> None:
    petri = Petri(n_wafer=2, ttime=5, takt_cycle=[0, 0])
    petri.reset()
    ok = petri.search()

    assert ok is True
    assert len(petri.full_transition_records) > 0
    for item in petri.full_transition_records:
        assert "transition" in item
        assert "fire_time" in item
        assert isinstance(item["transition"], str)
        assert isinstance(item["fire_time"], int)
