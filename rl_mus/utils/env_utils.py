import subprocess


def get_git_hash() -> str:
    return (
        subprocess.check_output(["git", "rev-parse", "--short", "HEAD"])
        .decode("ascii")
        .strip()
    )
