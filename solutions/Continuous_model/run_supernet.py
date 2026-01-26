from construct_net.super_net import ModuleSpec, RobotSpec, SuperPetriBuilder
from solutions.model.pn_models import Place, BasedToken
from collections import deque

import json
import numpy as np



def save_petri_split(result: dict, prefix: str):
    np.savez_compressed(
        f"{prefix}.npz",
        m0=result["m0"], md=result["md"],
        pre=result["pre"], pst=result["pst"],
        ptime=result["ptime"], ttime=result["ttime"],
        capacity=result["capacity"],
        n_wafer=result["n_wafer"],
    )

    marks_serialized = []
    for place in result["marks"]:
        tokens_data = []
        for token in place.tokens:
            tokens_data.append({
                "type": "BasedToken",
                "enter_time": token.enter_time
            })
        marks_serialized.append({
            "name": place.name,
            "capacity": place.capacity,
            "processing_time": place.processing_time,
            "type": place.type,
            "tokens": tokens_data
        })

    meta = {
        "edges": result["edges"],
        "id2p_name": result["id2p_name"],
        "id2t_name": result["id2t_name"],
        "idle_idx": result["idle_idx"],
        "nodes": result["nodes"],
        "module_x": {k: [v[0].tolist(), v[1].tolist()] for k, v in result["module_x"].items()},
        "marks": marks_serialized,
    }
    with open(f"{prefix}.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

def load_petri_split(prefix: str) -> dict:
    z = np.load(r"C:\Users\khand\OneDrive\code\dqn\CT\construct_net\petri_N8.npz.npz")
    with open(r"C:\Users\khand\OneDrive\code\dqn\CT\construct_net\petri_N8.npz.json", "r", encoding="utf-8") as f:
        meta = json.load(f)
    meta["module_x"] = {k: (np.array(v[0], dtype=int), np.array(v[1], dtype=int))
                        for k, v in meta["module_x"].items()}
    
    marks = []
    for place_data in meta["marks"]:
        tokens = deque()
        for token_data in place_data["tokens"]:
            tokens.append(BasedToken(
                enter_time=token_data["enter_time"]
            ))
        marks.append(Place(
            name=place_data["name"],
            capacity=place_data["capacity"],
            processing_time=place_data["processing_time"],
            type=place_data["type"],
            tokens=tokens
        ))
    meta["marks"] = marks
    
    return {**{k: z[k] for k in z.files}, **meta}


if __name__ == "__main__":
    # 模块库所：ptime 你手动填；capacity 你手动填（PM=1，LL可=2，LP按需）
    modules = {
        "LP": ModuleSpec(tokens=25, ptime=0, capacity=50),
        "AL":  ModuleSpec(tokens=0,  ptime=8, capacity=1),
        "LP_done": ModuleSpec(tokens=0, ptime=0, capacity=75),
        "LLA": ModuleSpec(tokens=0, ptime=20, capacity=2),
        "LLB": ModuleSpec(tokens=0, ptime=20, capacity=2),
        "LLC": ModuleSpec(tokens=0, ptime=0, capacity=1),
        "LLD": ModuleSpec(tokens=0, ptime=70, capacity=1),
        "PM2": ModuleSpec(tokens=0, ptime=70, capacity=2),
        "PM3": ModuleSpec(tokens=0, ptime=200, capacity=2),
        "PM1": ModuleSpec(tokens=0, ptime=600, capacity=4),
    }
    robots = {
        "TM1": RobotSpec(tokens=1, reach={"LP1", "LP2", "AL", "LLA", "LLB", "LP_done"}),
        "TM2": RobotSpec(tokens=2, reach={"LLA", "LLB", "LLC", "LLD", "PM7", "PM9",}),
        "TM3": RobotSpec(tokens=2, reach={"LLC", "LLD", "PM1", }),
    }

    # 多路径（含分叉）：示例：LP1->AL->(LLB)->(PM7/PM8)->LLD->(PM1/PM2)->LLA->LP_done
    # 这里用 LLB(进来) 和 LLA(出去)，LLD(进来) 和 LLC(出去) 由你路线来表达
    routes = [
        ["LP1", "AL", "LLA", ["PM7",], "LLC", ["PM1", ], "LLD", ["PM9"], "LLB", "LP_done"],     #路径C
    ]

    builder = SuperPetriBuilder(d_ptime=3, default_ttime=2)
    info = builder.build(modules, robots, routes)

    print("P,T =", info["pre"].shape[0], info["pre"].shape[1])
    print("edges =", len(info["edges"]))
    print("ptime len =", len(info["ptime"]), "ttime len =", len(info["ttime"]))
    print("capacity len =", len(info["capacity"]))

    save_petri_split(info, "petri_N8.npz")

