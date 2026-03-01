import math

# ---------- required k's per paper text ----------
def compute_ks(w, n):
    """
    w = number of expanded coefficients (paper's d)
    n = ring degree (paper's n)
    Returns k_i = n/2^i + 1 for i=0..log2(w)-1
    """
    assert (w & (w - 1)) == 0, "w must be a power of two"
    assert (n & (n - 1)) == 0, "n must be a power of two"
    levels = int(math.log2(w))
    return [n // (2 ** i) + 1 for i in range(levels)]

# ---------- subgroup table: build exponent map for <g> ----------
def build_subgroup_exp_map(g, n):
    """
    Build a dict mapping value -> smallest exponent u such that g^u ≡ value (mod 2n),
    by simulating powers of g until it cycles back to 1.
    """
    mod = 2 * n
    g %= mod
    if g % 2 == 0:
        return None  # not in Z*_{2n}

    exp = {}
    cur = 1
    u = 0
    while True:
        if cur not in exp:
            exp[cur] = u
        cur = (cur * g) % mod
        u += 1
        if cur == 1:
            break
        if u > mod:  # safety
            break
    return exp

def compute_u_list_totals_from_map(exp_map, w, n, verbose=True):
    """
    Returns (ks, u_list, tot_rot, tot_path)
      tot_rot  = sum_i u_i * 2^i   (total rotations across all intermediate nodes)
      tot_path = sum_i u_i         (sum of u_i's; your path objective)
    """
    ks = compute_ks(w, n)
    u_list = []
    tot_rot = 0
    tot_path = 0

    for i, k in enumerate(ks):
        if k not in exp_map:
            return None  # this g cannot generate all required substitutions

        u_i = exp_map[k]
        u_list.append(u_i)

        tot_path += u_i
        tot_rot += u_i * (2 ** i)

        if verbose:
            print(
                f"i={i:2d}, k={k:5d}, u_i={u_i:5d}, "
                f"u_i*2^i={u_i*(2**i):6d}"
            )

    return ks, u_list, tot_rot, tot_path

# ---------- brute-force best generator ----------
def min_tot_rot(w, n, verbose=False, progress_every=256):
    """
    Minimize tot_rot = sum_i u_i * 2^i.
    Returns (best_tot_rot, best_g, best_u_list, best_tot_path)
    """
    mod = 2 * n
    best = None  # (best_tot_rot, best_g, best_u_list, best_tot_path)

    for idx, g in enumerate(range(1, mod, 2), start=1):
        exp_map = build_subgroup_exp_map(g, n)
        if exp_map is None:
            continue

        res = compute_u_list_totals_from_map(exp_map, w, n, verbose=False)
        if res is None:
            continue
        ks, u_list, tot_rot, tot_path = res

        if best is None or tot_rot < best[0]:
            best = (tot_rot, g, u_list, tot_path)
            if verbose:
                print(f"[new best tot_rot] g={g}, tot_rot={tot_rot}, tot_path={tot_path}, u_list={u_list}")

        if progress_every and (idx % progress_every == 0):
            cur = None if best is None else (best[1], best[0])
            print(f"checked {idx}/{mod//2} candidates... current best (tot_rot): {cur}")

    return best

def min_tot_path(w, n, verbose=False, progress_every=256):
    """
    Minimize tot_path = sum_i u_i.
    Returns (best_tot_path, best_g, best_u_list, best_tot_rot)
    """
    mod = 2 * n
    best = None  # (best_tot_path, best_g, best_u_list, best_tot_rot)

    for idx, g in enumerate(range(1, mod, 2), start=1):
        exp_map = build_subgroup_exp_map(g, n)
        if exp_map is None:
            continue

        res = compute_u_list_totals_from_map(exp_map, w, n, verbose=False)
        if res is None:
            continue
        ks, u_list, tot_rot, tot_path = res

        if best is None or tot_path < best[0]:
            best = (tot_path, g, u_list, tot_rot)
            if verbose:
                print(f"[new best tot_path] g={g}, tot_path={tot_path}, tot_rot={tot_rot}, u_list={u_list}")

        if progress_every and (idx % progress_every == 0):
            cur = None if best is None else (best[1], best[0])
            print(f"checked {idx}/{mod//2} candidates... current best (tot_path): {cur}")

    return best

# ---------- main ----------
if __name__ == "__main__":
    w = 512
    n = 2048
    g = 5

    print(f"Using n={n}, w={w}, modulus=2n={2*n}, generator g={g}")
    print("Required k's:", compute_ks(w, n))
    print()

    exp_map = build_subgroup_exp_map(g, n)
    res = compute_u_list_totals_from_map(exp_map, w, n, verbose=True)
    if res is None:
        print("The chosen g cannot generate all required k_i; skipping summary.")
    else:
        ks, u_list, tot_rot, tot_path = res
        print(f"\nSummary for g = {g}:")
        print("\tu_list   =", u_list)
        print("\ttot_rot  =", tot_rot)
        print("\ttot_path =", tot_path)

    print("\n==============minimize tot_rot = sum u_i*2^i===================")
    best = min_tot_rot(w, n, verbose=False, progress_every=512)
    if best is None:
        print("No generator found that can generate all required k_i.")
    else:
        best_tot_rot, min_rot_g, best_u_list, best_tot_path = best
        print("\tmin_tot_rot_g =", min_rot_g)
        print("\tbest_u_list   =", best_u_list)
        print("\tbest_tot_rot  =", best_tot_rot)
        print("\t(best_tot_path for that g) =", best_tot_path)

    print("\n==============minimize tot_path = sum u_i===================")
    best2 = min_tot_path(w, n, verbose=False, progress_every=512)
    if best2 is None:
        print("No generator found that can generate all required k_i.")
    else:
        best_tot_path, min_path_g, best_u_list2, best_tot_rot2 = best2
        print("\tmin_tot_path_g =", min_path_g)
        print("\tbest_u_list    =", best_u_list2)
        print("\tbest_tot_path  =", best_tot_path)
        print("\t(best_tot_rot for that g) =", best_tot_rot2)