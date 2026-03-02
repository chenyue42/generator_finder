#!/usr/bin/env python3
# two_gen_bfs.py
#
# Implements the "two keys only" model:
# - allowed moves: multiply by g OR multiply by h (mod 2n)
# - no inverse moves unless you explicitly include them (we do NOT).
#
# It can:
#   1) Evaluate a given (g,h): coverage, u_list, tot_rot
#   2) Find best single generator (min tot_rot) by brute force
#   3) Heuristically find a good/best pair (g,h) by:
#        - brute-forcing all single g, keeping top-M
#        - brute-forcing all pairs among those top-M using BFS

import math
import argparse
from collections import deque

# ---------- required k's ----------
def compute_ks(w, n):
    """
    w = # expanded coefficients (paper's d)
    n = ring degree (paper's n)
    k_i = n/2^i + 1 for i=0..log2(w)-1
    """
    assert (w & (w - 1)) == 0, "w must be a power of two"
    assert (n & (n - 1)) == 0, "n must be a power of two"
    levels = int(math.log2(w))
    return [n // (2 ** i) + 1 for i in range(levels)]

# ---------- fast mapping: odd residue x in {1,3,...,2n-1} <-> idx in {0..n-1} ----------
def odd_to_idx(x):   # x must be odd
    return x >> 1

def idx_to_odd(i):
    return (i << 1) | 1

# ---------- BFS over directed Cayley graph with moves ×g and ×h ----------
def bfs_dist_to_targets(g, h, n, targets):
    """
    Graph vertices: odd residues mod 2n (size n).
    Edges: x -> x*g mod 2n, x -> x*h mod 2n.
    BFS from 1 until all targets reached.

    Returns: dist_target dict {k: distance} for reached targets (subset if unreachable).
    """
    mod = 2 * n
    g %= mod
    h %= mod
    # must be units; since mod is power-of-two, "odd" <=> unit
    if (g & 1) == 0 or (h & 1) == 0:
        return {}

    # normalize targets mod 2n
    targets = [t % mod for t in targets]
    target_set = set(targets)

    dist = [-1] * n
    q = deque()

    start = 1
    dist[odd_to_idx(start)] = 0
    q.append(start)

    found = {}
    if start in target_set:
        found[start] = 0

    while q and len(found) < len(target_set):
        x = q.popleft()
        dx = dist[odd_to_idx(x)]

        y1 = (x * g) % mod
        i1 = odd_to_idx(y1)
        if dist[i1] == -1:
            dist[i1] = dx + 1
            q.append(y1)
            if y1 in target_set:
                found[y1] = dx + 1

        y2 = (x * h) % mod
        i2 = odd_to_idx(y2)
        if dist[i2] == -1:
            dist[i2] = dx + 1
            q.append(y2)
            if y2 in target_set:
                found[y2] = dx + 1

    return found


def bfs_with_parents(g, h, n, targets):
    """
    BFS on odd residues mod 2n with moves ×g and ×h.
    Returns:
      dist (list[int]) over odd residues (size n), -1 if unreachable
      parent_idx (list[int]) predecessor node index, -1 for start/unreached
      parent_move (list[int]) 0 => used g, 1 => used h
    """
    mod = 2 * n
    g %= mod
    h %= mod
    if (g & 1) == 0 or (h & 1) == 0:
        return None, None, None

    targets = [t % mod for t in targets]
    target_set = set(targets)

    dist = [-1] * n
    parent_idx = [-1] * n
    parent_move = [-1] * n  # 0=g, 1=h
    q = deque()

    start = 1
    sidx = odd_to_idx(start)
    dist[sidx] = 0
    q.append(start)

    found_cnt = 1 if start in target_set else 0

    while q and found_cnt < len(target_set):
        x = q.popleft()
        dx = dist[odd_to_idx(x)]

        # move by g
        y = (x * g) % mod
        yi = odd_to_idx(y)
        if dist[yi] == -1:
            dist[yi] = dx + 1
            parent_idx[yi] = odd_to_idx(x)
            parent_move[yi] = 0
            q.append(y)
            if y in target_set:
                found_cnt += 1

        # move by h
        y = (x * h) % mod
        yi = odd_to_idx(y)
        if dist[yi] == -1:
            dist[yi] = dx + 1
            parent_idx[yi] = odd_to_idx(x)
            parent_move[yi] = 1
            q.append(y)
            if y in target_set:
                found_cnt += 1

    return dist, parent_idx, parent_move

def reconstruct_path(k, n, parent_idx, parent_move, g, h):
    """
    Reconstruct a shortest path (as a sequence of nodes and moves) from 1 to k.
    Returns:
      nodes: [1, ..., k]
      moves: ['g'/'h'] of length len(nodes)-1, aligned with edges between nodes
    """
    mod = 2 * n
    g %= mod
    h %= mod

    k %= mod
    ki = odd_to_idx(k)
    if ki < 0 or ki >= n or parent_idx[ki] == -1 and k != 1:
        return None, None

    # backtrack indices
    idxs = [ki]
    moves = []
    while idxs[-1] != odd_to_idx(1):
        cur = idxs[-1]
        pm = parent_move[cur]
        pi = parent_idx[cur]
        if pi == -1:
            return None, None
        moves.append('g' if pm == 0 else 'h')
        idxs.append(pi)

    idxs.reverse()
    moves.reverse()

    nodes = [idx_to_odd(i) for i in idxs]
    return nodes, moves

def eval_pair(g, h, w, n, verbose=True, show_paths=True):
    """
    Print u_list + tot_rot, and optionally print the actual shortest paths.
    """
    ks = compute_ks(w, n)
    dist, parent_idx, parent_move = bfs_with_parents(g, h, n, ks)

    u_list = []
    ok = True
    tot_rot = 0

    if dist is None:
        ok = False
        u_list = [None] * len(ks)
        if verbose:
            print(f"Invalid generators: g={g}, h={h} (must be odd mod 2n)")
        return ok, u_list, tot_rot

    if verbose:
        print(f"Pair (g={g}, h={h}) in Z*_{2*n} (mod {2*n})")
        print("ks:", ks)

    for i, k in enumerate(ks):
        di = dist[odd_to_idx(k % (2*n))]
        if di == -1:
            ok = False
            u_list.append(None)
            if verbose:
                print(f"  i={i:2d}  k={k:5d}  u_i=  None  (unreachable)")
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
        mod = 2 * n
        print("\n--- shortest paths (one example per k_i) ---")
        for i, k in enumerate(ks):
            if u_list[i] is None:
                continue
            nodes, moves = reconstruct_path(k, n, parent_idx, parent_move, g, h)
            if nodes is None:
                print(f"k={k}: (failed to reconstruct)")
                continue

            # Make a compact "word" like gghhg...
            word = "".join(moves)

            # Also show the node sequence (can be long; ok for small n)
            # For large n, you may prefer to print only the word.
            print(f"i={i:2d}, k={k:5d}, u_i={u_list[i]:3d}: word={word}")
            # Uncomment if you want the actual visited residues too:
            # print("    nodes:", " -> ".join(map(str, nodes)))

    return ok, u_list, tot_rot



# ---------- single-generator baseline (matches your previous tot_rot objective) ----------
def best_single_generator(w, n, progress_every=0):
    """
    Brute-force all odd g in [1..2n-1] and compute tot_rot for single generator
    where u_i is the discrete log in <g> (i.e., reachability using only ×g moves).
    For powers of two, every odd g is a unit.

    This is just eval_pair(g, g, ...) with BFS, but we can reuse the same BFS routine.
    """
    mod = 2 * n
    ks = compute_ks(w, n)

    best = None  # (tot_rot, g, u_list)

    for idx, g in enumerate(range(1, mod, 2), start=1):
        found = bfs_dist_to_targets(g, g, n, ks)
        ok = all(k in found for k in ks)
        if not ok:
            continue
        u_list = [found[k] for k in ks]
        tot_rot = sum(u_list[i] * (2 ** i) for i in range(len(u_list)))

        if best is None or tot_rot < best[0]:
            best = (tot_rot, g, u_list)

        if progress_every and (idx % progress_every == 0):
            print(f"[single] checked {idx}/{mod//2}... best so far: {None if best is None else (best[1], best[0])}")

    return best  # (tot_rot, g, u_list) or None

# ---------- two-generator search (heuristic but effective) ----------
def best_pair_from_top_singles(w, n, top_m=10, progress_every=0):
    """
    Practical search for medium n (e.g., 2048/4096):
      1) brute-force all single g to get tot_rot and keep top_m best g's
      2) brute-force all ordered pairs (g,h) among those top_m using BFS
         and pick best tot_rot
    """
    mod = 2 * n
    ks = compute_ks(w, n)

    # 1) rank singles
    singles = []  # list of (tot_rot, g, u_list)
    for idx, g in enumerate(range(1, mod, 2), start=1):
        found = bfs_dist_to_targets(g, g, n, ks)
        if not all(k in found for k in ks):
            continue
        u_list = [found[k] for k in ks]
        tot_rot = sum(u_list[i] * (2 ** i) for i in range(len(u_list)))
        singles.append((tot_rot, g, u_list))
        if progress_every and (idx % progress_every == 0):
            print(f"[rank singles] checked {idx}/{mod//2}... collected={len(singles)}")

    singles.sort(key=lambda x: x[0])
    top = singles[:top_m]
    cand = [g for (_, g, _) in top]

    if not cand:
        return None

    # 2) brute-force pairs among candidates
    best = None  # (tot_rot, g, h, u_list)
    total_pairs = len(cand) * len(cand)
    pair_cnt = 0

    for g in cand:
        for h in cand:
            pair_cnt += 1
            found = bfs_dist_to_targets(g, h, n, ks)
            if not all(k in found for k in ks):
                continue
            u_list = [found[k] for k in ks]
            tot_rot = sum(u_list[i] * (2 ** i) for i in range(len(u_list)))

            if best is None or tot_rot < best[0]:
                best = (tot_rot, g, h, u_list)

            if progress_every and (pair_cnt % progress_every == 0):
                print(f"[pairs] {pair_cnt}/{total_pairs}... best so far: {None if best is None else (best[1], best[2], best[0])}")

    return best, top  # best tuple + the top singles list (for reference)

# ---------- CLI ----------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=2048, help="ring degree n (power of two)")
    ap.add_argument("--w", type=int, default=512, help="expanded coeff count w (power of two)")
    ap.add_argument("--g", type=int, default=-1, help="generator g (odd)")
    ap.add_argument("--h", type=int, default=-1, help="generator h (odd)")
    ap.add_argument("--top_m", type=int, default=10, help="top-M singles used for pair search")
    ap.add_argument("--progress", type=int, default=0, help="print progress every K iterations (0 disables)")
    ap.add_argument("--search_pair", action="store_true", help="search best pair among top-M singles")
    ap.add_argument("--search_single", action="store_true", help="search best single generator")
    args = ap.parse_args()

    n = args.n
    w = args.w

    if args.g == -1 or args.h == -1:
        print("No generators provided, skipping pair evaluation. Use --g and --h to evaluate a specific pair.")
    else:
        print("\n======================== Evaluate given pair ==============================")
        eval_pair(args.g, args.h, w, n, verbose=True)
        print("=================================================================================\n")

    if args.search_single:
        print("=== Best single generator (min tot_rot) ===")
        best = best_single_generator(w, n, progress_every=args.progress)
        if best is None:
            print("No single generator found that covers all ks (unexpected for typical params).")
        else:
            tot_rot, g, u_list = best
            print(f"best_single_g={g}")
            print(f"u_list={u_list}")
            print(f"tot_rot={tot_rot}")
        print()

    if args.search_pair:
        print(f"=== Search best pair among top-{args.top_m} singles ===")
        res = best_pair_from_top_singles(w, n, top_m=args.top_m, progress_every=args.progress)
        if res is None:
            print("No candidates found.")
            return
        best_pair, top = res
        if best_pair is None:
            print("No pair among candidates covered all ks.")
            return
        tot_rot, g, h, u_list = best_pair
        print(f"best_pair_g={g}, best_pair_h={h}")
        print(f"u_list={u_list}")
        print(f"tot_rot={tot_rot}")
        print(f"\nSearched top {min(20, args.top_m)} singles (tot_rot, g) for reference:")
        for (t, gg, _) in top[:min(len(top), min(20, args.top_m))]:
            print(f"  {t:8d}  g={gg}")

if __name__ == "__main__":
    main()