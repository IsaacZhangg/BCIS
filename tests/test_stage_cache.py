"""Tests for stage_cache module."""

import numpy as np

from src.stage_cache import StageCache, source_content_hash


def test_source_content_hash_is_deterministic(tmp_path):
    path = tmp_path / "data.csv"
    path.write_bytes(b"some,data\n1,2\n")
    h1 = source_content_hash(path)
    h2 = source_content_hash(path)
    assert h1 == h2
    assert len(h1) == 16


def test_source_content_hash_detects_change(tmp_path):
    path = tmp_path / "data.csv"
    path.write_bytes(b"version a")
    h_a = source_content_hash(path)
    path.write_bytes(b"version b")
    h_b = source_content_hash(path)
    assert h_a != h_b


def test_get_returns_none_on_miss(tmp_path):
    cache = StageCache(cache_dir=tmp_path / "cache", config_hash="abc123")
    assert cache.get("preprocessed", "dead") is None


def test_put_then_get_roundtrip(tmp_path):
    cache = StageCache(cache_dir=tmp_path / "cache", config_hash="abc123")
    arr1 = np.arange(12).reshape(3, 4).astype(np.float32)
    arr2 = np.arange(5).astype(np.int64)
    cache.put("preprocessed", "s0", data=arr1, labels=arr2)

    loaded = cache.get("preprocessed", "s0")
    assert loaded is not None
    np.testing.assert_array_equal(loaded["data"], arr1)
    np.testing.assert_array_equal(loaded["labels"], arr2)


def test_different_config_hash_misses(tmp_path):
    cache_a = StageCache(cache_dir=tmp_path / "cache", config_hash="aaa")
    cache_b = StageCache(cache_dir=tmp_path / "cache", config_hash="bbb")
    cache_a.put("preprocessed", "s0", data=np.zeros(3))
    assert cache_b.get("preprocessed", "s0") is None


def test_different_source_hash_misses(tmp_path):
    cache = StageCache(cache_dir=tmp_path / "cache", config_hash="abc")
    cache.put("preprocessed", "old_hash", data=np.zeros(3))
    assert cache.get("preprocessed", "new_hash") is None
