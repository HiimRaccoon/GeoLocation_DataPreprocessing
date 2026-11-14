def test_launcher_sets_environment_before_paired_subprocesses(tmp_path, monkeypatch):
    import os
    import subprocess
    import geoguessr_benchmark.experiment as experiment
    monkeypatch.setattr(experiment,"verify_stage",lambda root,stage:{"stage":"initialized"})
    calls=[]
    def run(command,env,cwd):
        calls.append((command,env,cwd))
        return type("Result",(),{"returncode":0})()
    monkeypatch.setattr(subprocess,"run",run)
    assert experiment.train_all(tmp_path)
    assert [(cmd[cmd.index("--condition")+1],int(cmd[cmd.index("--seed")+1])) for cmd,env,cwd in calls] == [("original",42),("augmented",42),("original",123),("augmented",123),("original",2026),("augmented",2026)]
    for command,env,cwd in calls:
        assert env["PYTHONHASHSEED"] == command[command.index("--seed")+1]
        assert env["CUBLAS_WORKSPACE_CONFIG"] == ":4096:8"
