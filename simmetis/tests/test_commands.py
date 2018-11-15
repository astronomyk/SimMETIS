import pytest

def test_load_UserCommands_with_no_arguments():
    import simmetis as sim
    with pytest.raises(ValueError):
        cmd = sim.UserCommands()

def test_load_UserCommands_with_invalid_sim_data_dir():
    import simmetis as sim
    with pytest.raises(FileNotFoundError):
        cmd = sim.UserCommands(sim_data_dir="/")

# TODO Provide a config file that works?
def test_update_UserCommands():
    import simmetis as sim
    cmd = sim.UserCommands()
    cmd.update({'OBS_EXPTIME' : 30})
    assert cmd.cmds['OBS_EXPTIME'] == 30
    with pytest.raises(KeyError):
        cmd.update({'NO_EXISTE' : 30})
