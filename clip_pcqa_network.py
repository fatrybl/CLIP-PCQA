"""The fork's CLIP package and the CLIP-PCQA network, loaded from their files under names of
their own: clip and models are common top-level names (MM-PCQA's checkout holds a models
package too), so neither may enter a host process's sys.modules. The network imports clip by
bare name; it finds the fork's package under that name only while its file is executed."""

import sys
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
from types import ModuleType

ROOT = Path(__file__).parent


def own_module(name: str, path: Path) -> ModuleType:
    """The fork's module at path, loaded under name; a package's __init__.py keeps its
    relative imports.

    Args:
        name: Name the module is loaded under.
        path: Its file.
    """
    package = [str(path.parent)] if path.name == "__init__.py" else None
    spec = spec_from_file_location(name, path, submodule_search_locations=package)
    module = module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def network_module() -> ModuleType:
    """models/CLIPPCQA_Net.py, its bare imports of clip answered by the fork's package, and
    the names clip and clip.simple_tokenizer restored afterwards."""
    clip = own_module("clip_pcqa_clip", ROOT / "clip" / "__init__.py")
    aliases = {
        "clip": clip,
        "clip.simple_tokenizer": sys.modules["clip_pcqa_clip.simple_tokenizer"],
    }
    saved = {name: sys.modules.get(name) for name in aliases}
    sys.modules.update(aliases)
    try:
        network = own_module("clip_pcqa_net", ROOT / "models" / "CLIPPCQA_Net.py")
    finally:
        for name, module in saved.items():
            if module is None:
                del sys.modules[name]
            else:
                sys.modules[name] = module
    return network


NETWORK = network_module()
