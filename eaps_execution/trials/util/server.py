from pathlib import Path

from eta_nexus.servers.loaders.opcua_server_loader import load_opcua_servers_from_config

# absolute path of current file
current_file = Path(__file__).resolve()

# directory of current file
current_dir = current_file.parent

# one level up
parent_dir = current_dir.parent

# target path
config_path = parent_dir / "environments" / "connection.toml"


if __name__ == "__main__":
    servers = load_opcua_servers_from_config(config_path)
