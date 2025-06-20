import json
import os
from typing import Any

from hrl_tl.wrappers.utils import generate_all_specifications

if __name__ == "__main__":
    predicates: list[str] = [
        "psi_ld",
        "psi_bd",
        "psi_td",
        "psi_rd",
        "psi_gl",
        "psi_lv",
        "psi_hl",
    ]
    num_processes: int = 128
    specification_save_path: str = "out/maze/all_formulae.json"

    print(f"Generating all specifications for {len(predicates)} predicates...")
    specifications: list[str] = generate_all_specifications(predicates, num_processes)

    print(f"Saving specifications to {specification_save_path}...")
    saved_data: dict[str, Any] = {
        "predicates": predicates,
        "num_specifications": len(specifications),
        "specifications": specifications,
    }
    os.makedirs(os.path.dirname(specification_save_path), exist_ok=True)
    with open(specification_save_path, "w") as f:
        json.dump(saved_data, f, indent=4)
