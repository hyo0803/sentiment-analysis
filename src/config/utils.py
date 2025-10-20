import yaml
from pathlib import Path
from typing import Any, Dict, Optional, Union

# NOTE: use Optional[Union[Path, str]] instead of Path | str for compatibility with Python < 3.10
def load_config(config_path: Optional[Union[Path, str]] = None) -> Dict[str, Any]:
    """
    Загружает конфигурацию из YAML (config/params.yaml по умолчанию).
    Возвращает словарь или пустой dict при отсутствии/ошибке.
    """
    cfg_path = Path(config_path) if config_path else (Path(__file__).parent.parent / "config" / "params.yaml")
    if cfg_path.exists():
        try:
            return yaml.safe_load(cfg_path.read_text()) or {}
        except Exception:
            return {}
    return {}
