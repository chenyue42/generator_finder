import math
from collections import deque
from itertools import combinations

# ---------- required k's ----------
def compute_ks(w, n):
    assert (w & (w - 1)) == 0
    assert (n & (n - 1)) == 0
    levels = int(math.log2(w))
    return [n // (2 ** i) + 1 for i in range(levels)]

# ---------- odd residue <-> index ----------
def odd_to_idx(x):  # x must be odd
    return x >> 1

def idx_to_odd(i):
    return (i << 1) | 1

# ---------- BFS with parents for many generators ----------
def bfs_with_parents_multi(gens, n, targets):
    """
    gens: list[int] of generators (each step is multiply by gens[j] mod 2n)
    n: ring degree (modulus is 2n)
    targets: list[int] (odd residues mod 2n)

    Returns:
      dist         : list[int] size n, dist[idx] = shortest steps to that odd residue, -1 if unreachable
      parent_idx   : list[int] size n, predecessor index, -1 for start/unreached
      parent_move  : list[int] size n, which generator index (0..len(gens)-1) was used to get here
    """
    mod = 2 * n
    gens = [(g % mod) for g in gens]

    # power-of-two modulus => unit iff odd
    if any((g & 1) == 0 for g in gens):
        return None, None, None

    targets = [t % mod for t in targets]
    target_set = set(targets)

    dist = [-1] * n
    parent_idx = [-1] * n
    parent_move = [-1] * n

    q = deque()
    start = 1
    sidx = odd_to_idx(start)
    dist[sidx] = 0
    q.append(start)

    found_cnt = 1 if start in target_set else 0

    while q and found_cnt < len(target_set):
        x = q.popleft()
        dx = dist[odd_to_idx(x)]

        for j, g in enumerate(gens):
            y = (x * g) % mod
            yi = odd_to_idx(y)
            if dist[yi] == -1:
                dist[yi] = dx + 1
                parent_idx[yi] = odd_to_idx(x)
                parent_move[yi] = j
                q.append(y)
                if y in target_set:
                    found_cnt += 1

    return dist, parent_idx, parent_move

def reconstruct_path_multi(k, n, parent_idx, parent_move, gen_labels=None):
    mod = 2 * n
    k %= mod
    ki = odd_to_idx(k)

    if k != 1 and parent_idx[ki] == -1:
        return None, None

    idxs = [ki]
    moves = []

    while idxs[-1] != odd_to_idx(1):
        cur = idxs[-1]
        pm = parent_move[cur]
        pi = parent_idx[cur]
        if pi == -1:
            return None, None
        moves.append(pm)
        idxs.append(pi)

    idxs.reverse()
    moves.reverse()

    nodes = [idx_to_odd(i) for i in idxs]
    if gen_labels is None:
        move_labels = [f"g{j}" for j in moves]
    else:
        move_labels = [gen_labels[j] for j in moves]
    return nodes, move_labels

def eval_gens(gens, w, n, verbose=True, show_paths=True, gen_labels=None, show_nodes=False):
    """
    gens: list[int] generators (odd ints mod 2n)
    Returns: (ok, u_list, tot_rot)
    """
    ks = compute_ks(w, n)
    dist, parent_idx, parent_move = bfs_with_parents_multi(gens, n, ks)

    u_list = []
    ok = True
    tot_rot = 0

    if dist is None:
        if verbose:
            print("Invalid generator list: all generators must be odd mod 2n.")
        return False, [None] * len(ks), 0

    mod = 2 * n
    if verbose:
        labels = gen_labels if gen_labels is not None else [f"g{i}" for i in range(len(gens))]
        print(f"Generators: {list(zip(labels, gens))}  (mod {mod})")
        print("ks:", ks)

    for i, k in enumerate(ks):
        di = dist[odd_to_idx(k % mod)]
        if di == -1:
            ok = False
            u_list.append(None)
            if verbose:
                print(f"  i={i:2d}  k={k:5d}  u_i=None  (unreachable)")
        else:
            u_i = di
            u_list.append(u_i)
            tot_rot += u_i * (2 ** i)
            if verbose:
                print(f"  i={i:2d}  k={k:5d}  u_i={u_i:5d}  u_i*2^i={u_i*(2**i):6d}")

    if verbose:
        print("ok     =", ok)
        print("u_list =", u_list)
        print("tot_rot=", tot_rot)

    if verbose and show_paths:
        print("\n--- shortest paths (one example per k_i) ---")
        for i, k in enumerate(ks):
            if u_list[i] is None:
                continue
            nodes, move_labels = reconstruct_path_multi(k, n, parent_idx, parent_move, gen_labels=gen_labels)
            word = " ".join(move_labels)
            print(f"i={i:2d}, k={k:5d}, u_i={u_list[i]:3d}: word={word}")
            if show_nodes:
                print("    nodes:", " -> ".join(map(str, nodes)))

    return ok, u_list, tot_rot

# ---------- auto selection ----------
def candidate_generators(n, cap=None, exclude_one=True):
    """
    Default candidate set: all odd residues mod 2n (units).
    For large n you can cap to keep it fast.
    """
    mod = 2 * n
    cands = [g for g in range(1, mod, 2)]
    if exclude_one:
        cands = [g for g in cands if g != 1]
    if cap is not None:
        cands = cands[:cap]
    return cands

def score_gens(gens, w, n):
    """
    Return (ok, u_list, tot_rot) without printing.
    """
    ok, u_list, tot_rot = eval_gens(gens, w, n, verbose=False, show_paths=False)
    return ok, u_list, tot_rot

def auto_select_gens(w, n, t=2, method="greedy", cap=None, top_m=None, progress_every=0):
    """
    Choose a generator set (size t) to minimize tot_rot.

    method:
      - "greedy": forward selection (fast, good for n=2048/4096)
      - "bruteforce": exact over combinations (only feasible with small cap)

    Parameters:
      cap: limit candidate generator list size (recommended for bruteforce)
      top_m: for greedy, optionally pre-rank singles and only keep top_m as pool
      progress_every: print progress for long runs
    """
    cands = candidate_generators(n, cap=cap)

    # Optional: restrict pool by best singles
    if top_m is not None:
        singles = []
        for idx, g in enumerate(cands, start=1):
            ok, u_list, tot_rot = score_gens([g], w, n)
            if ok:
                singles.append((tot_rot, g))
            if progress_every and (idx % progress_every == 0):
                print(f"[rank singles] {idx}/{len(cands)} collected={len(singles)}")
        singles.sort()
        cands = [g for _, g in singles[:top_m]]
        if not cands:
            return None

    if method == "bruteforce":
        best = None  # (tot_rot, gens, u_list)
        combos = list(combinations(cands, t))
        for idx, gens in enumerate(combos, start=1):
            ok, u_list, tot_rot = score_gens(list(gens), w, n)
            if ok and (best is None or tot_rot < best[0]):
                best = (tot_rot, list(gens), u_list)
            if progress_every and (idx % progress_every == 0):
                cur = None if best is None else (best[1], best[0])
                print(f"[bruteforce] {idx}/{len(combos)} best={cur}")
        return best

    if method == "greedy":
        chosen = []
        best_state = None  # (tot_rot, chosen, u_list)

        pool = cands[:]
        for step in range(t):
            best_step = None  # (tot_rot, g, u_list)
            for idx, g in enumerate(pool, start=1):
                if g in chosen:
                    continue
                trial = chosen + [g]
                ok, u_list, tot_rot = score_gens(trial, w, n)
                if not ok:
                    continue
                if best_step is None or tot_rot < best_step[0]:
                    best_step = (tot_rot, g, u_list)

                if progress_every and (idx % progress_every == 0):
                    print(f"[greedy step {step+1}] checked {idx}/{len(pool)}")

            if best_step is None:
                return None
            _, g_best, u_list_best = best_step
            chosen.append(g_best)
            best_state = (best_step[0], chosen[:], u_list_best)

        return best_state

    raise ValueError("method must be 'greedy' or 'bruteforce'")

# ---------- demo ----------
if __name__ == "__main__":
    n = 4096
    w = 512
    t = 3
    top_m = 300

    # print("=== Evaluate a fixed generator set ===")
    # eval_gens([5, 3], w, n, verbose=True, show_paths=False, gen_labels=["g=5", "h=3"])
    # print()

    print(f"=== Auto-select t={t} generators to minimize tot_rot (greedy) ===")
    best = auto_select_gens(w, n, t=t, method="greedy", top_m=top_m, progress_every=0)
    if best is None:
        print("No generator set found.")
    else:
        tot_rot, gens, u_list = best
        print("best_gens =", gens)
        print("u_list    =", u_list)
        print("tot_rot   =", tot_rot)
        # show details + paths
        labels = [f"g{i}={g}" for i, g in enumerate(gens)]
        eval_gens(gens, w, n, verbose=True, show_paths=True, gen_labels=labels, show_nodes=False)