import logging

import pytest
from pydantic import BaseModel, ValidationError

from src.utils import BaseModelSharing

# --- Test Models ---
# TODO: make it work


class SharedConfig(BaseModel):
    param_a: str = "default_a"
    param_b: int = 1


class ComponentA(BaseModel):
    shared: SharedConfig
    comp_a_param: str = "comp_a"


class ComponentB(BaseModel):
    shared: SharedConfig
    comp_b_param: float = 0.5


class MainModel(BaseModelSharing):
    _shared_fields_config = {"shared_config": ["component_a", "component_b"]}

    shared_config: SharedConfig
    component_a: ComponentA
    component_b: ComponentB
    other_param: str = "main"


class ModelWithNonBaseModelShared(BaseModelSharing):
    _shared_fields_config = {"shared_list": ["comp_a"]}
    shared_list: list  # Not a BaseModel
    comp_a: ComponentA  # Depends on a non-BaseModel, should warn/skip


class ModelWithBadConfig(BaseModelSharing):
    _shared_fields_config = {"non_existent_shared": ["comp_a"]}  # Config error
    comp_a: ComponentA


class ModelWithBadDependentConfig(BaseModelSharing):
    _shared_fields_config = {"shared_config": ["non_existent_comp"]}  # Config error
    shared_config: SharedConfig


# --- Test Cases ---


def test_instantiation_with_dicts():
    """Test standard instantiation using dictionaries."""
    data = {
        "shared_config": {"param_a": "test_a", "param_b": 10},
        "component_a": {"comp_a_param": "a1"},
        "component_b": {"comp_b_param": 2.0},
        "other_param": "main_test",
    }
    model = MainModel(**data)

    assert model.shared_config.param_a == "test_a"
    assert model.shared_config.param_b == 10
    assert model.component_a.comp_a_param == "a1"
    assert model.component_b.comp_b_param == 2.0
    assert model.other_param == "main_test"
    # Check if the shared instance is the same object
    assert model.shared_config is model.component_a.shared
    assert model.shared_config is model.component_b.shared


def test_instantiation_with_shared_instance():
    """Test instantiation providing a pre-existing shared instance."""
    pre_shared = SharedConfig(param_a="pre_a", param_b=20)
    data = {
        "shared_config": pre_shared,
        "component_a": {"comp_a_param": "a2"},
        "component_b": {"comp_b_param": 3.0},
    }
    model = MainModel(**data)

    assert model.shared_config is pre_shared  # Should be the exact same instance
    assert model.shared_config.param_a == "pre_a"
    assert model.shared_config.param_b == 20
    assert model.component_a.comp_a_param == "a2"
    assert model.component_b.comp_b_param == 3.0
    # Check if the shared instance is the same object
    assert model.shared_config is model.component_a.shared
    assert model.shared_config is model.component_b.shared


def test_instantiation_with_defaults():
    """Test instantiation when shared/dependent data is missing."""
    model = MainModel()  # No data provided

    assert model.shared_config.param_a == "default_a"
    assert model.shared_config.param_b == 1
    assert model.component_a.comp_a_param == "comp_a"
    assert model.component_b.comp_b_param == 0.5
    assert model.other_param == "main"
    # Check if the shared instance is the same object
    assert model.shared_config is model.component_a.shared
    assert model.shared_config is model.component_b.shared


def test_instantiation_missing_shared_dict():
    """Test when shared config dict is missing, should use defaults."""
    data = {
        # "shared_config": {...}, # Missing
        "component_a": {"comp_a_param": "a3"},
        "component_b": {"comp_b_param": 4.0},
    }
    model = MainModel(**data)

    assert model.shared_config.param_a == "default_a"  # Default used
    assert model.shared_config.param_b == 1  # Default used
    assert model.component_a.comp_a_param == "a3"
    assert model.component_b.comp_b_param == 4.0
    assert model.shared_config is model.component_a.shared
    assert model.shared_config is model.component_b.shared


def test_instantiation_missing_dependent_dict():
    """Test when dependent config dict is missing, should use defaults."""
    data = {
        "shared_config": {"param_a": "test_a4"},
        # "component_a": {...}, # Missing
        "component_b": {"comp_b_param": 5.0},
    }
    model = MainModel(**data)

    assert model.shared_config.param_a == "test_a4"
    assert model.component_a.comp_a_param == "comp_a"  # Default used
    assert model.component_b.comp_b_param == 5.0
    assert model.shared_config is model.component_a.shared
    assert model.shared_config is model.component_b.shared


def test_redundant_shared_in_dependent(caplog):
    """Test warning when shared config is defined redundantly in dependent."""
    data = {
        "shared_config": {"param_a": "top_level", "param_b": 100},
        "component_a": {
            "shared": {"param_a": "redundant_a", "param_b": 999},  # Redundant
            "comp_a_param": "a5",
        },
        "component_b": {"comp_b_param": 6.0},
    }
    with caplog.at_level(logging.WARNING):
        model = MainModel(**data)

    assert "defined redundantly in 'component_a' data" in caplog.text
    assert model.shared_config.param_a == "top_level"  # Top level wins
    assert model.shared_config.param_b == 100  # Top level wins
    assert model.component_a.comp_a_param == "a5"
    assert model.component_b.comp_b_param == 6.0
    assert model.shared_config is model.component_a.shared  # Still shared correctly
    assert model.shared_config is model.component_b.shared


def test_invalid_data_type_for_shared(caplog):
    """Test warning when data for shared field is not dict or instance."""
    data = {
        "shared_config": "not a dict or instance",  # Invalid type
        "component_a": {},
        "component_b": {},
    }
    with caplog.at_level(logging.WARNING):
        model = MainModel(**data)

    assert "is neither a valid instance nor a dict. Using defaults." in caplog.text
    # Should fall back to defaults for the shared config
    assert model.shared_config.param_a == "default_a"
    assert model.shared_config.param_b == 1
    assert model.shared_config is model.component_a.shared
    assert model.shared_config is model.component_b.shared


def test_config_error_shared_field_not_in_model():
    """Test ValueError when shared field in config doesn't exist in model."""
    data = {"comp_a": {}}
    with pytest.raises(
        ValueError,
        match="Config Error: Shared field 'non_existent_shared' not in ModelWithBadConfig",
    ):
        ModelWithBadConfig(**data)


def test_config_warning_dependent_field_not_in_model(caplog):
    """Test Warning when dependent field in config doesn't exist in model."""
    data = {"shared_config": {}}
    with caplog.at_level(logging.WARNING):
        ModelWithBadDependentConfig(
            **data
        )  # Instantiation should still work, just skip bad dependent
    assert "Config Warning: Dependent field 'non_existent_comp' not found" in caplog.text


def test_shared_field_not_basemodel(caplog):
    """Test warning when configured shared field is not a BaseModel subclass."""
    data = {"shared_list": [1, 2], "comp_a": {}}  # comp_a expects SharedConfig
    with caplog.at_level(logging.WARNING):
        # This will likely fail standard Pydantic validation later,
        # but the injector should log a warning and skip injection.
        with pytest.raises(
            ValidationError
        ):  # Expect validation error because comp_a won't get a SharedConfig
            ModelWithNonBaseModelShared(**data)

    assert (
        "Skipping injection: Shared field 'shared_list' is not a BaseModel subclass."
        in caplog.text
    )


def test_dependent_data_is_instance_raises_error():
    """Test ValueError when dependent data is already an instance."""
    pre_shared = SharedConfig(param_a="pre_shared")
    # Create an instance that would normally be a dependent field's value
    pre_comp_a = ComponentA(shared=pre_shared, comp_a_param="pre_comp_a")

    data = {
        "shared_config": pre_shared,  # Provide top-level shared instance
        "component_a": pre_comp_a,  # Provide pre-built component A instance
        "component_b": {},
    }

    # Expect a ValueError because component_a is already an instance
    with pytest.raises(
        ValueError,
        match=r"Cannot inject shared field 'shared_config'.*dependent field 'component_a' is already an instance",
    ):
        MainModel(**data)
