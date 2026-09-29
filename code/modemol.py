import copy
import os
import sys
from itertools import combinations
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.special import comb

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

TFVAE_ROOT = os.environ.get("TFVAE_ROOT", "/home/gcb20242750/pythonwork/tf-vae/TransformerVAE-main")
TFVAE_CONFIG_PATH = os.environ.get(
    "TFVAE_CONFIG_PATH",
    os.path.join(TFVAE_ROOT, "training/results/tfvae_qedsa_100ep_20260719_r14_5m_bs8k_epochloss/config.yaml"),
)
TFVAE_R6_WEIGHT_PATH = os.environ.get(
    "TFVAE_R6_WEIGHT_PATH",
    os.path.join(
        TFVAE_ROOT,
        "training/results/contrastive_finetune_gsk_jnk_20260723_r6_continue_ep20_to_ep25/models/epoch_3",
    ),
)
TFVAE_HIGH_QUALITY_DATA = os.environ.get(
    "TFVAE_HIGH_QUALITY_DATA",
    "/home/gcb20242750/pythonwork/cddd_contrastive_modemol/data/finetune_gsk_jnk/finetune_train_enriched_chembl_gsk_jnk_315k_max320.csv",
)
TFVAE_DECODE_MAX_LEN = int(os.environ.get("TFVAE_DECODE_MAX_LEN", "120"))
TFVAE_BATCH_SIZE = int(os.environ.get("TFVAE_BATCH_SIZE", "128"))

_tfvae_bundle = None
_high_quality_smiles_cache = None


def _get_tfvae_bundle():
    global _tfvae_bundle
    if _tfvae_bundle is not None:
        return _tfvae_bundle
    if TFVAE_ROOT not in sys.path:
        sys.path.insert(0, TFVAE_ROOT)
    import torch
    import yaml
    from addict import Dict as AddictDict
    from src import Model
    from src.datasets.tokenizer import VocabularyTokenizer

    with open(TFVAE_CONFIG_PATH, "rb") as handle:
        tfvae_config = AddictDict(yaml.load(handle, yaml.Loader))
    with open(os.path.join(TFVAE_ROOT, "data/smiles_vocs.txt"), encoding="utf-8") as handle:
        tokenizer = VocabularyTokenizer(handle.read().splitlines())

    device_name = os.environ.get("TFVAE_DEVICE", "")
    if device_name:
        device = torch.device(device_name)
    else:
        device = torch.device("cuda", 0) if torch.cuda.is_available() else torch.device("cpu")

    model_logger = type("TFVAELogger", (), {"debug": lambda *a, **k: None, "warning": lambda *a, **k: None})()
    model = Model(logger=model_logger, **copy.deepcopy(tfvae_config.model))
    model.load(path=TFVAE_R6_WEIGHT_PATH, strict=True)
    model.to(device)
    model.eval()
    _tfvae_bundle = {"model": model, "tokenizer": tokenizer, "device": device}
    return _tfvae_bundle


def _pad_token_batch(token_lists, pad_token, device):
    import torch

    max_len = max(len(x) for x in token_lists)
    batch = torch.full((len(token_lists), max_len), pad_token, dtype=torch.long, device=device)
    for i, toks in enumerate(token_lists):
        batch[i, : len(toks)] = torch.tensor(toks, dtype=torch.long, device=device)
    return batch


def encode_molecules(smiles_list: List[str], batch_size: int = TFVAE_BATCH_SIZE) -> np.ndarray:
    import torch

    bundle = _get_tfvae_bundle()
    model = bundle["model"]
    tokenizer = bundle["tokenizer"]
    device = bundle["device"]
    token_lists = [tokenizer.tokenize(str(smi)) for smi in smiles_list]
    mus = []
    with torch.no_grad():
        for start in range(0, len(token_lists), batch_size):
            inp = _pad_token_batch(token_lists[start : start + batch_size], tokenizer.pad_token, device)
            mask = model["masker"](inp)
            emb = model["enc_embedding"](inp)
            memory = model["encoder"](emb, mask)
            latent_base = model["pooler"](memory, torch.transpose(mask, 0, 1))
            mu = model["latent2mu"](latent_base)
            mus.append(mu.detach().cpu())
    return torch.cat(mus, dim=0).numpy().astype(np.float64)


def _greedy_decode_batch(latents, max_len: int = TFVAE_DECODE_MAX_LEN) -> List[str]:
    import torch

    bundle = _get_tfvae_bundle()
    model = bundle["model"]
    tokenizer = bundle["tokenizer"]
    device = bundle["device"]
    latents = torch.as_tensor(latents, dtype=torch.float32, device=device)
    batch_size = latents.shape[0]
    state = model["decoder"](latent=latents, mode="prepare_cell_forward")
    cur_input, outs = model["dec_supporter"](batch_size=batch_size, mode="init")
    cur_input = cur_input.to(device)
    with torch.no_grad():
        for pos in range(max_len):
            cur_emb = model["dec_embedding"](cur_input, position=pos)
            cur_output, state = model["decoder"](
                tgt=cur_emb,
                latent=latents,
                state=state,
                position=pos,
                mode="cell_forward",
            )
            cur_output = model["dec2proba"](cur_output)
            cur_input, outs = model["dec_supporter"](cur_proba=cur_output, outs=outs, mode="add")
            if torch.all(cur_input.squeeze(1) == tokenizer.end_token):
                break
        toks = model["dec_supporter"](outs, mode="aggregate").detach().cpu().numpy()
    return [tokenizer.detokenize(row.tolist()) for row in toks]


def decode_molecules(population: np.ndarray, batch_size: int = TFVAE_BATCH_SIZE) -> List[str]:
    if population.ndim == 1:
        population = population.reshape(1, -1)
    smiles_list = []
    for start in range(0, population.shape[0], batch_size):
        batch = population[start : start + batch_size]
        try:
            smiles_list.extend(_greedy_decode_batch(batch))
        except Exception:
            smiles_list.extend([""] * len(batch))
    return smiles_list


def load_high_quality_smiles(
    n_needed: int,
    data_csv: str = TFVAE_HIGH_QUALITY_DATA,
    top_pool_multiplier: int = 20,
) -> List[str]:
    global _high_quality_smiles_cache
    if _high_quality_smiles_cache is None:
        df = pd.read_csv(data_csv)
        smi_col = "canonical_smiles" if "canonical_smiles" in df.columns else "smiles"
        if smi_col not in df.columns:
            smi_col = "source_smiles"
        df = df.dropna(subset=[smi_col, "gsk", "jnk"]).copy()
        df["gsk"] = pd.to_numeric(df["gsk"], errors="coerce")
        df["jnk"] = pd.to_numeric(df["jnk"], errors="coerce")
        df = df.dropna(subset=["gsk", "jnk"])
        df["combined_gsk_jnk"] = (df["gsk"] + df["jnk"]) / 2.0
        df = df.sort_values(["combined_gsk_jnk", "gsk", "jnk"], ascending=False)
        _high_quality_smiles_cache = df[smi_col].astype(str).drop_duplicates().tolist()

    pool_size = min(len(_high_quality_smiles_cache), max(n_needed * top_pool_multiplier, n_needed))
    pool = _high_quality_smiles_cache[:pool_size]
    if len(pool) < n_needed:
        raise ValueError(f"high-quality pool too small: {len(pool)} < {n_needed}")
    indices = np.random.choice(len(pool), size=n_needed, replace=False)
    return [pool[int(i)] for i in indices]


def get_objectives(
    population: np.ndarray,
    objectives_list: list,
    reference_smiles: Optional[str] = None,
) -> Tuple[np.ndarray, List[str]]:
    from get_objectives import _calculate_molecular_objective

    if population.ndim == 1:
        population = population.reshape(1, -1)
    smiles_list = decode_molecules(population)
    objectives = np.array(
        [
            _calculate_molecular_objective(smi, objectives_list, reference_smiles=reference_smiles)
            for smi in smiles_list
        ]
    )
    return objectives, smiles_list


def uniform_weight_vectors(N: int, M: int) -> np.ndarray:
    if M == 1:
        return np.ones((N, 1))
    H1 = 1
    try:
        while comb(H1 + M - 1, M - 1, exact=True) <= N:
            H1 += 1
        H1 -= 1
    except Exception:
        H1 = max(1, N // M)
    W = np.zeros((0, M))
    if H1 >= 1:
        try:
            indices = list(combinations(range(H1 + M - 1), M - 1))
            if indices:
                W = np.array(indices) - np.tile(np.array(range(M - 1)), (len(indices), 1))
                W = np.hstack((W, H1 + np.zeros((W.shape[0], 1)))) - np.hstack(
                    (np.zeros((W.shape[0], 1)), W)
                )
                W = W / H1 if H1 > 0 else W
        except Exception:
            pass
    if W.shape[0] < N:
        if W.shape[0] == 0:
            W = np.random.rand(N, M)
        else:
            W_add = np.random.rand(N - W.shape[0], M)
            W = np.vstack((W, W_add))
        W = W / W.sum(axis=1, keepdims=True)
    elif W.shape[0] > N:
        indices = np.random.choice(W.shape[0], N, replace=False)
        W = W[indices]
    W[W < 1e-6] = 1e-6
    W = W / W.sum(axis=1, keepdims=True)
    return W


def cosine_distance(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    x_norm = np.sqrt(np.sum(x**2, axis=1, keepdims=True))
    y_norm = np.sqrt(np.sum(y**2, axis=1, keepdims=True))
    cosine_sim = np.dot(x, y.T) / (x_norm * y_norm.T + 1e-10)
    return 1 - cosine_sim


def scalarized_fitness(
    objectives: np.ndarray,
    weight: np.ndarray,
    z_star: np.ndarray,
    maximize: bool = True,
) -> float:
    diff = z_star - objectives if maximize else objectives - z_star
    return float(np.dot(diff, weight))


def is_dominated(obj1: np.ndarray, obj2: np.ndarray, maximize: bool = True) -> bool:
    if maximize:
        return np.all(obj2 >= obj1) and np.any(obj2 > obj1)
    return np.all(obj2 <= obj1) and np.any(obj2 < obj1)


def crossover(
    parent1: np.ndarray,
    parent2: np.ndarray,
    d: float = 0.5,
    bounds: Tuple[float, float] = (0.0, 1.0),
) -> np.ndarray:
    u = np.random.rand()
    r = -d + (1 + 2 * d) * u
    offspring = parent1 + r * (parent2 - parent1)
    return np.clip(offspring, bounds[0], bounds[1])


def mutation(
    individual: np.ndarray,
    pm: float = 0.1,
    sigma: float = 0.1,
    bounds: Tuple[float, float] = (0.0, 1.0),
) -> np.ndarray:
    mutated = individual.copy()
    dim = len(individual)
    num_segments = 32
    segment_size = dim // num_segments
    for i in range(num_segments):
        start_idx = i * segment_size
        end_idx = start_idx + segment_size
        if np.random.rand() < pm:
            mutated[start_idx:end_idx] += np.random.randn(segment_size) * sigma
    return np.clip(mutated, bounds[0], bounds[1])


def generate_offspring(
    mating_pool: list,
    pc: float = 1.0,
    pm: float = 0.3,
    d: float = 0.5,
    sigma: float = 0.3,
    bounds: Tuple[float, float] = (0.0, 1.0),
) -> np.ndarray:
    pool_size = len(mating_pool)
    if pool_size < 2:
        return mutation(mating_pool[0], pm, sigma, bounds)
    idx1, idx2 = np.random.choice(pool_size, 2, replace=False)
    parent1, parent2 = mating_pool[idx1], mating_pool[idx2]
    if np.random.rand() < pc:
        offspring = crossover(parent1, parent2, d, bounds)
    else:
        offspring = parent1.copy()
    return mutation(offspring, pm, sigma, bounds)


def calculate_neighbors(W: np.ndarray, T: int) -> np.ndarray:
    N = W.shape[0]
    distances = cosine_distance(W, W)
    neighbors = np.zeros((N, T), dtype=int)
    for i in range(N):
        dist_i = distances[i].copy()
        dist_i[i] = np.inf
        neighbors[i] = np.argsort(dist_i)[:T]
    return neighbors


def update_reference_point(
    z_star: np.ndarray,
    objectives: np.ndarray,
    maximize: bool = True,
    is_adaptive: bool = False,
) -> np.ndarray:
    if not np.isfinite(objectives).all() or not is_adaptive:
        return z_star
    if maximize:
        return np.maximum(z_star, objectives)
    return np.minimum(z_star, objectives)


def add_to_archive(
    archive_pop: list,
    archive_obj: list,
    individual: np.ndarray,
    objectives: np.ndarray,
    maximize: bool = True,
) -> Tuple[list, list]:
    if not np.isfinite(objectives).all():
        return archive_pop, archive_obj
    for obj_archived in archive_obj:
        if np.array_equal(objectives, obj_archived):
            return archive_pop, archive_obj
    for obj_archived in archive_obj:
        if is_dominated(objectives, obj_archived, maximize):
            return archive_pop, archive_obj
    indices_to_remove = []
    for i, obj_archived in enumerate(archive_obj):
        if is_dominated(obj_archived, objectives, maximize):
            indices_to_remove.append(i)
    for idx in reversed(indices_to_remove):
        archive_pop.pop(idx)
        archive_obj.pop(idx)
    archive_pop.append(individual.copy())
    archive_obj.append(objectives.copy())
    return archive_pop, archive_obj


def maintain_archive(archive_pop: list, archive_obj: list, max_size: int) -> Tuple[list, list]:
    if len(archive_pop) <= max_size:
        return archive_pop, archive_obj
    pop_array = np.array(archive_pop)
    obj_array = np.array(archive_obj)
    from nonDominationSort import nonDominationSort, crowdingDistanceSort

    ranks = nonDominationSort(pop_array, obj_array)
    distances = crowdingDistanceSort(pop_array, obj_array, ranks)
    indices = np.arange(len(archive_pop))
    sorted_indices = sorted(indices, key=lambda i: (ranks[i], -distances[i]))
    selected_indices = sorted_indices[:max_size]
    return [archive_pop[i] for i in selected_indices], [archive_obj[i] for i in selected_indices]


def compute_fingerprint_from_smiles(smiles: str):
    if not smiles or not isinstance(smiles, str):
        return None
    from rdkit import Chem
    from rdkit.Chem import AllChem

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    return AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048)


def dynamic_prune_archive(
    archive_pop: list,
    archive_obj: list,
    gen: int,
    Genmax: int,
    delta_max: np.ndarray,
    delta_min: np.ndarray,
    maximize: bool = True,
) -> Tuple[list, list]:
    if len(archive_obj) < 3:
        return archive_pop, archive_obj
    obj_array = np.array(archive_obj)
    if maximize:
        z_ideal = np.max(obj_array, axis=0)
        z_nadir = np.min(obj_array, axis=0)
    else:
        z_ideal = np.min(obj_array, axis=0)
        z_nadir = np.max(obj_array, axis=0)
    denom = np.maximum(np.abs(z_ideal - z_nadir), 1e-6)
    norm_obj = np.abs(obj_array - z_ideal) / denom
    delta_t = ((delta_min - delta_max) / Genmax) * gen + delta_max
    keep_pop = []
    keep_obj = []
    for i, norm_v in enumerate(norm_obj):
        if np.any(norm_v > delta_t):
            continue
        keep_pop.append(archive_pop[i])
        keep_obj.append(archive_obj[i])
    if len(keep_obj) == 0:
        return archive_pop, archive_obj
    return keep_pop, keep_obj


def identify_shortboard_and_select_parent(
    current_obj: np.ndarray,
    w: np.ndarray,
    archive_pop: list,
    archive_obj: list,
    z_ideal: np.ndarray,
    z_nadir: np.ndarray,
    maximize: bool = True,
    K: int = 5,
):
    if len(archive_obj) == 0:
        return None
    denom = np.maximum(np.abs(z_ideal - z_nadir), 1e-6)
    norm_current = np.abs(current_obj - z_ideal) / denom
    w_safe = np.maximum(w, 1e-6)
    gap = norm_current / w_safe
    m_star = np.argmax(gap)
    obj_array = np.array(archive_obj)
    if maximize:
        sorted_indices = np.argsort(-obj_array[:, m_star])
    else:
        sorted_indices = np.argsort(obj_array[:, m_star])
    top_k_indices = sorted_indices[:K]
    selected_idx = np.random.choice(top_k_indices)
    return archive_pop[selected_idx]


def modemol_algorithm(
    objective_func: Callable[[np.ndarray], Tuple[np.ndarray, List[str]]],
    N: int = 100,
    NA: int = 200,
    T: int = 10,
    Genmax: int = 100,
    n_variables: int = 512,
    bounds: Tuple[float, float] = (-1.0, 1.0),
    maximize: bool = True,
    pc: float = 0.9,
    pm: float = 0.3,
    d: float = 0.5,
    sigma: float = 0.3,
    delta: float = 0.9,
    z_star_fix_ratio: float = 0.6,
    shortboard_interval: int = 10,
    initial_population: Optional[np.ndarray] = None,
    initial_objectives: Optional[np.ndarray] = None,
    initial_smiles: Optional[List[str]] = None,
    objectives_list: Optional[list] = None,
    reference_smiles: Optional[str] = None,
    threshold_z_star: Optional[np.ndarray] = None,
    delta_max: Optional[np.ndarray] = None,
    delta_min: Optional[np.ndarray] = None,
    use_div: bool = True,
    use_archieve: bool = True,
    use_dynamic_resource: bool = True,
    resource_interval: int = 20,
    n_active: int = 100,
    epsilon: float = 1e-4,
    lambda_max: float = 2.0,
    gamma: float = 1.0,
) -> Dict:
    if objectives_list is None:
        objectives_list = ["qed", "sa", "jnk3", "similarity"]
    if initial_population is not None and initial_objectives is not None and initial_smiles is not None:
        population = initial_population.copy()
        objectives = initial_objectives.copy()
        init_smiles = initial_smiles.copy()
    else:
        population = np.random.uniform(bounds[0], bounds[1], (N, n_variables))
        objectives, init_smiles = objective_func(
            population,
            objectives_list,
            reference_smiles=reference_smiles,
        )

    n_objectives = len(objectives_list)
    weights = uniform_weight_vectors(N, n_objectives)
    if use_div:
        population_fps = [compute_fingerprint_from_smiles(smi) for smi in init_smiles]
    else:
        population_fps = [None] * len(init_smiles)

    if threshold_z_star is None:
        threshold_z_star = np.max(objectives, axis=0)
    else:
        threshold_z_star = np.asarray(threshold_z_star, dtype=np.float64)

    z_star = np.max(objectives, axis=0) * 1.25
    z_star = np.minimum(z_star, threshold_z_star)
    neighbors = calculate_neighbors(weights, T)
    archive_pop = []
    archive_obj = []
    for i in range(N):
        archive_pop, archive_obj = add_to_archive(
            archive_pop,
            archive_obj,
            population[i],
            objectives[i],
            maximize,
        )

    success_counters = np.zeros(weights.shape[0])
    active_indices = np.arange(weights.shape[0])
    z_star_switched = False

    for gen in range(Genmax):
        n_weights = weights.shape[0]
        gen_ratio = (gen + 1) / Genmax
        is_z_star_adaptive = gen_ratio < z_star_fix_ratio
        if not is_z_star_adaptive and not z_star_switched:
            z_star_switched = True
            z_star = threshold_z_star.copy()

        if use_dynamic_resource and gen > 0 and gen % resource_interval == 0:
            activity_scores = success_counters[:n_weights] / resource_interval + epsilon
            probabilities = activity_scores / np.sum(activity_scores)
            active_indices = np.random.choice(
                np.arange(n_weights),
                size=min(n_active, n_weights),
                replace=True,
                p=probabilities,
            )
            success_counters = np.zeros(n_weights)

        trigger_shortboard = use_archieve and ((gen + 1) % shortboard_interval == 0)
        if trigger_shortboard and len(archive_obj) > 0 and delta_max is not None and delta_min is not None:
            archive_pop, archive_obj = dynamic_prune_archive(
                archive_pop,
                archive_obj,
                gen,
                Genmax,
                delta_max,
                delta_min,
                maximize,
            )
            if len(archive_obj) > 0:
                obj_array_all = np.vstack((objectives, np.array(archive_obj)))
                if maximize:
                    z_ideal_global = np.max(obj_array_all, axis=0)
                    z_nadir_global = np.min(obj_array_all, axis=0)
                else:
                    z_ideal_global = np.min(obj_array_all, axis=0)
                    z_nadir_global = np.max(obj_array_all, axis=0)

        new_individuals = []
        mating_info = []
        target_indices = active_indices if use_dynamic_resource else np.arange(n_weights)

        for w_idx in target_indices:
            use_neighbor = False
            if trigger_shortboard and len(archive_obj) > 0 and delta_max is not None and delta_min is not None:
                parent_archive = identify_shortboard_and_select_parent(
                    objectives[w_idx],
                    weights[w_idx],
                    archive_pop,
                    archive_obj,
                    z_ideal_global,
                    z_nadir_global,
                    maximize,
                    K=min(T, len(archive_pop)),
                )
                if parent_archive is not None:
                    mating_pool = [population[w_idx], parent_archive]
                elif np.random.rand() < delta:
                    neighbor_indices = neighbors[w_idx]
                    mating_pool = [population[idx] for idx in neighbor_indices]
                else:
                    selected_indices = np.random.choice(N, size=min(T, N), replace=False)
                    mating_pool = [population[idx] for idx in selected_indices]
            elif np.random.rand() < delta:
                neighbor_indices = neighbors[w_idx]
                mating_pool = [population[idx] for idx in neighbor_indices]
            else:
                selected_indices = np.random.choice(N, size=min(T, N), replace=False)
                mating_pool = [population[idx] for idx in selected_indices]

            new_individual = generate_offspring(mating_pool, pc, pm, d, sigma, bounds)
            new_individuals.append(new_individual)
            mating_info.append({"w_idx": w_idx, "use_neighbor": np.random.rand() < delta})

        new_individuals_array = np.array(new_individuals)
        all_new_objectives, all_new_smiles = objective_func(
            new_individuals_array,
            objectives_list,
            reference_smiles=reference_smiles,
        )

        if use_div:
            all_new_fps = [compute_fingerprint_from_smiles(smi) for smi in all_new_smiles]
        else:
            all_new_fps = [None] * len(all_new_smiles)

        for idx, (new_individual, new_objectives) in enumerate(zip(new_individuals, all_new_objectives)):
            w_idx = mating_info[idx]["w_idx"]
            use_neighbor = mating_info[idx]["use_neighbor"]
            z_star = update_reference_point(z_star, new_objectives, maximize, is_z_star_adaptive)
            if use_neighbor:
                update_candidates = neighbors[w_idx].copy()
            else:
                update_candidates = np.arange(N)
            if w_idx not in update_candidates:
                update_candidates = np.append(update_candidates, w_idx)

            if use_div:
                from rdkit import DataStructs

                fp_new = all_new_fps[idx]

            for cand_w_idx in update_candidates:
                cand_weight = weights[cand_w_idx]
                old_scalar = scalarized_fitness(
                    objectives[cand_w_idx],
                    cand_weight,
                    z_star,
                    maximize,
                )
                new_scalar = scalarized_fitness(
                    new_objectives,
                    cand_weight,
                    z_star,
                    maximize,
                )

                if use_div:
                    lambda_iter = lambda_max * (((gen + 1) / Genmax) ** gamma)
                    S_j_indices = neighbors[cand_w_idx]
                    fp_old = population_fps[cand_w_idx]
                    max_sim_new = 0.0
                    max_sim_old = 0.0
                    if fp_new is not None and fp_old is not None:
                        for n_idx in S_j_indices:
                            fp_n = population_fps[n_idx]
                            if fp_n is not None:
                                sim_new = DataStructs.TanimotoSimilarity(fp_new, fp_n)
                                if sim_new > max_sim_new:
                                    max_sim_new = sim_new
                                if n_idx != cand_w_idx:
                                    sim_old = DataStructs.TanimotoSimilarity(fp_old, fp_n)
                                    if sim_old > max_sim_old:
                                        max_sim_old = sim_old
                    old_dist = old_scalar + lambda_iter * max_sim_old
                    new_dist = new_scalar + lambda_iter * max_sim_new
                else:
                    old_dist = old_scalar
                    new_dist = new_scalar

                if new_dist < old_dist:
                    population[cand_w_idx] = new_individual.copy()
                    objectives[cand_w_idx] = new_objectives.copy()
                    if cand_w_idx < len(success_counters):
                        success_counters[cand_w_idx] += 1
                    if use_div:
                        population_fps[cand_w_idx] = fp_new

            archive_pop, archive_obj = add_to_archive(
                archive_pop,
                archive_obj,
                new_individual,
                new_objectives,
                maximize,
            )

        archive_pop, archive_obj = maintain_archive(archive_pop, archive_obj, NA)

    return {
        "population": population,
        "objectives": objectives,
        "archive_population": np.array(archive_pop),
        "archive_objectives": np.array(archive_obj),
        "weights": weights,
        "reference_point": z_star,
        "generation": Genmax,
    }


def init_population(
    N: int,
    reference_smiles: str,
    sigma: float = 0.5,
    bounds: Tuple[float, float] = (-1.0, 1.0),
    objectives_list: Optional[list] = None,
    high_quality_data: str = TFVAE_HIGH_QUALITY_DATA,
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    if objectives_list is None:
        objectives_list = ["qed", "sa", "gsk3b", "similarity"]
    z_0 = encode_molecules([reference_smiles])
    if z_0 is None or z_0.shape[0] == 0:
        raise ValueError(f"无法编码分子: {reference_smiles}")

    high_n = N // 2
    perturb_n = N - high_n
    high_quality_smiles = load_high_quality_smiles(high_n, data_csv=high_quality_data)
    high_population = encode_molecules(high_quality_smiles)
    high_population = np.clip(high_population, bounds[0], bounds[1])

    gauss_noise = np.random.normal(0, 1, (perturb_n * 5, z_0.shape[1]))
    perturb_candidates = np.clip(gauss_noise * sigma + z_0, bounds[0], bounds[1])

    from nonDominationSort import nonDominationSort, crowdingDistanceSort

    perturb_objectives, perturb_smiles = get_objectives(
        perturb_candidates,
        objectives_list,
        reference_smiles=reference_smiles,
    )
    ranks = nonDominationSort(perturb_candidates, perturb_objectives)
    dis = crowdingDistanceSort(perturb_candidates, perturb_objectives, ranks)
    indices = np.arange(len(perturb_candidates))
    sorted_indices = sorted(indices, key=lambda i: (ranks[i], -dis[i]))
    selected = sorted_indices[:perturb_n]
    perturb_population = perturb_candidates[selected]
    perturb_objectives = perturb_objectives[selected]
    perturb_smiles = [perturb_smiles[i] for i in selected]

    high_objectives, high_decoded_smiles = get_objectives(
        high_population,
        objectives_list,
        reference_smiles=reference_smiles,
    )

    population = np.vstack([high_population, perturb_population])
    final_objectives = np.vstack([high_objectives, perturb_objectives])
    final_smiles = list(high_decoded_smiles) + list(perturb_smiles)
    return population, final_objectives, final_smiles


def optimize_single_molecule(reference_smiles: str, config: Dict, seed: Optional[int] = None) -> Dict:
    if seed is not None:
        np.random.seed(seed)

    initial_population, initial_objectives, initial_smiles = init_population(
        config["N"],
        reference_smiles,
        sigma=config["sigma"],
        bounds=tuple(config.get("bounds", (-5.0, 5.0))),
        objectives_list=config["objectives_list"],
        high_quality_data=config.get("tfvae_high_quality_data", TFVAE_HIGH_QUALITY_DATA),
    )

    threshold_z_star = np.array(
        [config.get("threshold_list", {}).get(obj_name, 0.0) for obj_name in config["objectives_list"]],
        dtype=np.float64,
    )
    delta_max = np.array(
        [config.get("delta_max_list", {}).get(obj, 0.8) for obj in config["objectives_list"]],
        dtype=np.float64,
    )
    delta_min = np.array(
        [config.get("delta_min_list", {}).get(obj, 0.3) for obj in config["objectives_list"]],
        dtype=np.float64,
    )

    result = modemol_algorithm(
        get_objectives,
        N=config["N"],
        NA=config["NA"],
        T=config.get("T", 10),
        Genmax=config["Genmax"],
        n_variables=config["n_variables"],
        maximize=config["maximize"],
        bounds=tuple(config.get("bounds", (-5.0, 5.0))),
        initial_population=initial_population,
        initial_objectives=initial_objectives,
        initial_smiles=initial_smiles,
        objectives_list=config["objectives_list"],
        reference_smiles=reference_smiles,
        pc=config["pc"],
        pm=config["pm"],
        d=config["d"],
        sigma=config["sigma"],
        delta=config["delta"],
        z_star_fix_ratio=config["z_star_fix_ratio"],
        shortboard_interval=config["shortboard_interval"],
        threshold_z_star=threshold_z_star,
        delta_max=delta_max,
        delta_min=delta_min,
        use_div=config.get("use_div", True),
        use_archieve=config.get("use_archieve", True),
        lambda_max=config.get("lambda_max", 2.0),
        gamma=config.get("gamma", 1.0),
    )

    archive_smiles = decode_molecules(result["archive_population"]) if len(result["archive_population"]) else []
    success_individuals = []
    for smiles, obj in zip(archive_smiles, result["archive_objectives"]):
        ok = True
        for i, obj_name in enumerate(config["objectives_list"]):
            threshold = config.get("threshold_list", {}).get(obj_name)
            if threshold is not None and obj[i] < threshold:
                ok = False
                break
        if ok:
            item = {name: float(obj[i]) for i, name in enumerate(config["objectives_list"])}
            item["smiles"] = smiles
            success_individuals.append(item)

    result["archive_smiles"] = archive_smiles
    result["success_individuals"] = success_individuals
    return result


if __name__ == "__main__":
    config = {
        "N": 100,
        "NA": 100,
        "T": 10,
        "n_variables": 512,
        "bounds": (-5.0, 5.0),
        "Genmax": 100,
        "maximize": True,
        "objectives_list": ["qed", "sa", "gsk3b", "sim"],
        "pc": 1.0,
        "pm": 0.5,
        "d": 0.5,
        "sigma": 0.5,
        "delta": 0.9,
        "z_star_fix_ratio": 1.0,
        "shortboard_interval": 10,
        "threshold_list": {"qed": 0.7, "sa": 0.7, "gsk3b": 0.4, "sim": 0.2},
        "delta_max_list": {"qed": 0.2, "sa": 0.2, "gsk3b": 0.1, "sim": 0.2},
        "delta_min_list": {"qed": 0.0, "sa": 0.0, "gsk3b": 0.0, "sim": 0.0},
        "data_file": "data_filter/gsk3_test_100.csv",
        "base_seed": 1,
        "tfvae_high_quality_data": TFVAE_HIGH_QUALITY_DATA,
        "use_div": True,
        "use_archieve": False,
        "lambda_max": 2.0,
        "gamma": 1.0,
    }

    data_file = os.path.join(current_dir, config["data_file"])
    df = pd.read_csv(data_file)
    smiles_col = next((c for c in df.columns if str(c).lower() == "smiles"), df.columns[0])
    for idx, reference_smiles in enumerate(df[smiles_col].astype(str)):
        result = optimize_single_molecule(
            reference_smiles,
            config,
            seed=config["base_seed"] + idx if config.get("base_seed") is not None else None,
        )
        print(idx, reference_smiles, len(result["archive_population"]), len(result["success_individuals"]))
