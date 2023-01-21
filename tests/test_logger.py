"""Unitary tests of loggers utilities"""
from typing import Dict, Any
import pytest
from rlpackage.loggers import create_logger

@pytest.mark.parametrize("config", [{"logger": {"logger": "tensorboard", "name":"test"}, "random": 1},
                                    {"logger": {"logger": "wandb", "name":"test"}, "random": 1}])
@pytest.mark.filterwarnings()
def test_create_logger(config: Dict[str, Any]):
    """test function"""
    logger = create_logger(config)
    for step in range(10):
        logger.log_step(step=step, info={"random": 10*step +5}, testing=False)
    logger.close_logger()
