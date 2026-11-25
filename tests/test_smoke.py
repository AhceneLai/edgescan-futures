import yaml


def test_config_loads():
    with open("config.yaml", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    assert {"contracts", "costs", "account", "setups"} <= set(cfg)


def test_package_imports():
    import edgescan.cli  # noqa: F401
