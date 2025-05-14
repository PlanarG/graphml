import docker
from pathlib import Path

import docker.types

client = docker.from_env()
current_dir = Path(__file__).parent

containers = []

cpu_groups = [2,4]

for i in cpu_groups:
    cpu_start = i * 4
    cpu_end = cpu_start + 3
    cpu_set = f"{cpu_start}-{cpu_end}"

    container = client.containers.run(
        image="graphml",
        detach=True,
        cpuset_cpus=cpu_set,
        volumes={
            current_dir / "all_logs": {"bind": "/logs", "mode": "rw"},
            current_dir / "tmp_workspace" : {"bind": "/workspace", "mode": "rw"},
            "/usr1/data/sijiel/mle-bench/data": {"bind": "/data", "mode": "rw"},
            current_dir / "test" / "run.sh": {"bind": "/run.sh", "mode": "ro"},
            current_dir / "test" / f"task{i}.txt": {"bind": "/tasks.txt", "mode": "rw"},
        },
        auto_remove=True,
        name=f"aide-{i}",
        command="/bin/bash /run.sh",
        device_requests=[
            docker.types.DeviceRequest(
                capabilities=[["gpu"]],
                device_ids=[str(i)],
            )
        ],
        shm_size="32g",
    )

    containers.append(container)
    print(f"Containers started, GPU {i} CPU {cpu_set}")
