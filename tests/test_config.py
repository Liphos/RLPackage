"""Unitary tests of config utilities"""
import pytest
from rlpackage.configs import load_config

@pytest.mark.parametrize("yaml_path, success",
                         [("configs/test_fail.yaml", False), ("configs/test_cartpole.yaml", True)])
def test_load_config(yaml_path:str, success:bool):
    """test load function"""
    if not success:
        with pytest.raises(ValueError) as exc_info:
            load_config(yaml_path)

        exception_raised = exc_info.value
        assert exception_raised
    else:
        configs = load_config(yaml_path)
        assert configs["env"]
        assert configs["replay_buffer"]
