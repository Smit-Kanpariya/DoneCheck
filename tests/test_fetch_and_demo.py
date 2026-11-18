"""The download (without the network) and the demo's example picking."""

import http.client

import pytest

from donecheck import demo
from donecheck import fetch as fetching


@pytest.fixture(autouse=True)
def no_waiting(monkeypatch):
    monkeypatch.setattr(fetching.time, "sleep", lambda s: None)


def test_download_retries_a_body_cut_short(tmp_path, monkeypatch):
    replies = [http.client.IncompleteRead(b"{"), b"{}"]
    monkeypatch.setattr(fetching, "get", lambda url: r if isinstance(r := replies.pop(0), bytes) else (_ for _ in ()).throw(r))
    path = tmp_path / "e" / "trajs" / "a.traj.json"
    assert fetching.download(("u", path, 2)) == "new"
    assert path.read_bytes() == b"{}" and not list(path.parent.glob("*.part"))
    assert fetching.download(("u", path, 2)) == "kept"


def test_download_gives_up_on_the_wrong_size(tmp_path, monkeypatch):
    monkeypatch.setattr(fetching, "get", lambda url: b"{}")
    with pytest.raises(RuntimeError, match="the listing says 5"):
        fetching.download(("u", tmp_path / "a.traj.json", 5))
    assert not (tmp_path / "a.traj.json").exists()


def test_jobs_keep_only_run_files_and_exit_statuses(tmp_path, monkeypatch):
    prefix = "bash-only/e/trajs/"
    monkeypatch.setattr(fetching, "listing", lambda p: [(prefix + "a__a-1/a__a-1.traj.json", 10),
                                                        (prefix + "exit_statuses_1.yaml", 3),
                                                        (prefix + "a__a-1/a__a-1.pred", 4), (prefix + "b/c.traj.json", 5)])
    jobs = fetching.jobs_for("e", tmp_path)
    assert [(p.relative_to(tmp_path).as_posix(), size) for _, p, size in jobs] == [
        ("e/metadata.yaml", None), ("e/per_instance_details.json", None),
        ("e/trajs/a__a-1.traj.json", 10), ("e/exit_statuses_1.yaml", 3)]


def test_main_checks_the_run_count_and_parses_every_run(tmp_path, monkeypatch):
    monkeypatch.setitem(fetching.ENTRIES, "test", ["e"])
    jobs = lambda entry, dest: [(f"u{i}", dest / entry / "trajs" / f"{i}.traj.json", None) for i in "ab"]  # noqa: E731
    monkeypatch.setattr(fetching, "jobs_for", jobs)
    monkeypatch.setattr(fetching, "get", lambda url: b"{}")
    fetching.main("test", str(tmp_path))
    (tmp_path / "e" / "trajs" / "b.traj.json").write_bytes(b"{")  # cut short, but the right count
    monkeypatch.setattr(fetching, "download", lambda job: "kept")
    with pytest.raises(ValueError):
        fetching.main("test", str(tmp_path))
    (tmp_path / "e" / "trajs" / "b.traj.json").unlink()
    with pytest.raises(SystemExit, match="1 run files on disk, but the listing has 2"):
        fetching.main("test", str(tmp_path))


def case(id, score, empty=False, tests=0):
    return {"id": id, "score": score, "facts": {"submission_empty": empty, "tests_after_last_edit": tests}}


def test_pick_skips_empty_groups_and_never_repeats_a_run():
    empty = {g: [] for g in ("caught", "missed", "false alarm", "accepted, fixed")}
    assert demo.pick(empty) == []
    cases = empty | {"missed": [case("m1", 0.2, tests=1)], "false alarm": [case(f"f{i}", 0.9) for i in range(5)]}
    assert [(g, c["id"]) for g, c in demo.pick(cases)] == [("missed", "m1"), ("false alarm", "f0"),
                                                          ("false alarm", "f1"), ("false alarm", "f2")]
