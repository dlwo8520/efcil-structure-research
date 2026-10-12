from selective.protocol import EXPECTED_PARTITIONS, canonical_hash, validate_partition
from selective.registry import REDUCED_PROTOCOLS, REDUCED_RUN_IDS, method_registry, resolve_method


def test_stage1_registry_is_exactly_27():
    registry = method_registry()
    assert len(registry) == 27
    assert list(registry)[:6] == ["C0", "A_P005", "A_P010", "A_P015", "A_P020", "A_P025"]
    assert "B_P010_A050" in registry
    assert "C_P010_L030" in registry


def test_reduced_stage1_registry_and_structure_overrides():
    assert REDUCED_PROTOCOLS == ("IA5", "CAR5")
    assert REDUCED_RUN_IDS == (
        "Structure", "A_P005", "B_P020_A100", "C_P020_L030", "C_P020_L100"
    )
    structure = resolve_method("Structure")
    assert structure["old_new_separation"] is False
    assert structure["adaptive_margin_mode"] == "none"
    assert structure["selective_method"] == "none"
    assert structure["similarity_relation_mode"] == "none"
    assert structure["similarity_scc_enabled"] is False
    assert structure["sep_lambda"] == 0.0
    assert structure["top_p"] is None
    assert structure["alpha"] is None
    assert structure["lambda_clad"] is None


def test_exact_partitions_cover_every_class():
    totals = {"IA5": 200, "IA10": 200, "CAR5": 196, "CAR10": 196}
    for protocol, total in totals.items():
        partition = validate_partition(protocol, EXPECTED_PARTITIONS[protocol], total)
        assert sum(partition) == total
        assert all(value > 0 for value in partition)
    assert EXPECTED_PARTITIONS["CAR5"] == [40, 39, 39, 39, 39]
    assert EXPECTED_PARTITIONS["CAR10"] == [20] * 6 + [19] * 4


def test_partition_hash_depends_on_boundaries():
    order = list(range(200))
    ia5 = {"sizes": [40] * 5, "class_ids_per_task": [order[i:i + 40] for i in range(0, 200, 40)]}
    ia10 = {"sizes": [20] * 10, "class_ids_per_task": [order[i:i + 20] for i in range(0, 200, 20)]}
    assert canonical_hash(ia5) != canonical_hash(ia10)
