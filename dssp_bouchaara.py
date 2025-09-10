#!/usr/bin/env python3
import argparse
import numpy as np
from Bio.PDB import PDBParser

# ---------- utils ----------
def norm(v):
    n = np.linalg.norm(v)
    return v / n if n != 0 else v

def dist(a, b):
    d = np.linalg.norm(a - b)
    return d if d > 1e-6 else 1e-6

# ---------- parse PDB ----------
def parse_structure(pdb_file, chain_id="A"):
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("prot", pdb_file)
    model = structure[0]
    if chain_id not in model:
        raise SystemExit(f"Chain {chain_id} not found in {pdb_file}")
    chain = model[chain_id]

    residues = []
    for res in chain:
        if res.id[0] != " ":
            continue  # skip HETATM/water
        atoms = {}
        for name in ("N", "CA", "C", "O", "H", "HN"):
            if name in res:
                atoms[name] = res[name].get_coord().astype(float)
        if all(k in atoms for k in ("N", "CA", "C", "O")):
            residues.append({
                "resseq": res.id[1],
                "resname": res.resname,
                "atoms": atoms
            })
    return residues

# ---------- place H (virtual if missing) ----------
def place_H_first(N, C, CA, r_NH=1.0):
    v1 = norm(C - N); v2 = norm(CA - N)
    bvec = v1 + v2
    b = norm(bvec) if np.linalg.norm(bvec) >= 1e-6 else -v1
    return N - b * r_NH

def place_H_prevC(prev_C, N, CA, r_NH=1.0):
    u = norm(prev_C - N)
    v = norm(CA - N)
    bvec = u + v
    b = norm(bvec) if np.linalg.norm(bvec) >= 1e-6 else -u
    return N - b * r_NH

def ensure_H_coords(residues):
    prev_C = None
    for r in residues:
        at = r["atoms"]
        # break chain if previous C is too far from current N
        if prev_C is not None:
            dCN = np.linalg.norm(prev_C - at["N"])
            if not (1.2 <= dCN <= 2.2):
                prev_C = None
        if "H" in at:
            prev_C = at["C"]
            continue
        if "HN" in at:
            at["H"] = at["HN"]
        else:
            if prev_C is not None:
                at["H"] = place_H_prevC(prev_C, at["N"], at["CA"])
            else:
                at["H"] = place_H_first(at["N"], at["C"], at["CA"])
        prev_C = at["C"]

# ---------- H-bond energy (DSSP-like) ----------
q1, q2, F = 0.42, 0.20, 332.0

def hb_energy(C, O, N, H):
    return q1 * q2 * (
        1.0/dist(O, N) + 1.0/dist(C, H) - 1.0/dist(O, H) - 1.0/dist(C, N)
    ) * F

def find_hbonds(residues, cutoff=-0.45):
    # returns dict (i,j) -> E for CO(i) -> NH(j)
    HB = {}
    n = len(residues)
    for i in range(n):
        C, O = residues[i]["atoms"]["C"], residues[i]["atoms"]["O"]
        for j in range(n):
            if abs(i - j) < 2:
                continue
            N, H = residues[j]["atoms"]["N"], residues[j]["atoms"]["H"]
            E = hb_energy(C, O, N, H)
            if E < cutoff:
                HB[(i, j)] = E
    return HB

# ---------- turns & helices ----------
def detect_turns(HB, n, L):
    # set of i such that CO(i) -> NH(i+n)
    return {i for (i, j) in HB.keys() if j == i + n and 0 <= i < L - n}

def assign_helices_from_turns(turns, L, span, letter):
    # Two consecutive n-turns => minimal helix i..i+span, extend by overlap
    state = np.array(["C"] * L, dtype="<U1")
    i = 1
    while i <= L - (span + 1):
        if (i - 1 in turns) and (i in turns):
            start, end = i, i + span
            k = i + 1
            while (k - 1 in turns) and (k in turns):
                end += 1
                k += 1
            state[start:end+1] = letter
            i = k
        else:
            i += 1
    return state

def assign_helices_all(HB, L):
    t3 = detect_turns(HB, 3, L)
    t4 = detect_turns(HB, 4, L)
    t5 = detect_turns(HB, 5, L)
    G = assign_helices_from_turns(t3, L, span=2, letter="G")
    H = assign_helices_from_turns(t4, L, span=3, letter="H")
    I = assign_helices_from_turns(t5, L, span=4, letter="I")
    return G, H, I, t3, t4, t5

# ---------- beta bridges / ladders ----------
def is_parallel_bridge(HB, i, j):
    return ((i-1, j) in HB and (j, i+1) in HB) or ((j-1, i) in HB and (i, j+1) in HB)

def is_antiparallel_bridge(HB, i, j):
    return ((i, j) in HB and (j, i) in HB) or ((i-1, j+1) in HB and (j-1, i+1) in HB)

def find_bridges(HB, L):
    bridges = []
    for i in range(1, L-1):
        for j in range(1, L-1):
            if abs(i - j) < 3:
                continue
            if is_parallel_bridge(HB, i, j):
                bridges.append((i, j, "P"))
            if is_antiparallel_bridge(HB, i, j):
                bridges.append((i, j, "A"))
    return bridges

def build_ladders(bridges):
    # group consecutive bridges of same type into ladders
    bridge_set = set(bridges)
    used = set()
    ladders = []
    for b in bridges:
        if b in used:
            continue
        i, j, t = b
        di, dj = (1, 1) if t == "P" else (1, -1)
        # forward
        fwd = []
        ii, jj = i, j
        while (ii, jj, t) in bridge_set and (ii, jj, t) not in used:
            fwd.append((ii, jj, t))
            used.add((ii, jj, t))
            ii += di
            jj += dj
        # backward
        bwd = []
        ii, jj = i - di, j - dj
        while (ii, jj, t) in bridge_set and (ii, jj, t) not in used:
            bwd.append((ii, jj, t))
            used.add((ii, jj, t))
            ii -= di
            jj -= dj
        ladder = list(reversed(bwd)) + fwd
        if ladder:
            ladders.append(ladder)
    return ladders

def assign_beta_states(bridges, L):
    ladders = build_ladders(bridges)
    stateB = np.array(["C"] * L, dtype="<U1")
    stateE = np.array(["C"] * L, dtype="<U1")
    for lad in ladders:
        residues_in_ladder = set()
        for (i, j, _) in lad:
            residues_in_ladder.add(i)
            residues_in_ladder.add(j)
        if len(lad) >= 2:
            for k in residues_in_ladder:
                stateE[k] = "E"
        else:
            for k in residues_in_ladder:
                stateB[k] = "B"
    return stateB, stateE, ladders

# ---------- non-helical turns (T) ----------
def assign_T_from_turns(t3, t4, t5, existing_states):
    L = len(existing_states)
    T = np.array(["C"] * L, dtype="<U1")
    for n, turns in ((3, t3), (4, t4), (5, t5)):
        for i in turns:
            for pos in (i, i + n):
                if 0 <= pos < L and existing_states[pos] == "C":
                    T[pos] = "T"
    return T

# ---------- merge with DSSP priority ----------
PRIORITY = {'H':7,'B':6,'E':5,'G':4,'I':3,'T':2,'S':1,'C':0}

def merge_states(*cols):
    L = len(cols[0])
    out = []
    for k in range(L):
        pick = 'C'; best = -1
        for col in cols:
            c = col[k]
            pr = PRIORITY.get(c, 0)
            if pr > best:
                best = pr
                pick = c
        out.append(pick)
    return np.array(out, dtype="<U1")

# ---------- printing helpers ----------
def format_summary(residues, state):
    s = "".join(state.tolist())
    idx = " ".join(f"{r['resseq']:>3d}" for r in residues)
    aa  = " ".join(f"{r['resname']:<3s}" for r in residues)
    return s, idx, aa

# ---------- main run ----------
def run(pdb, chain, cutoff, verbose=False, print_bridges=False, print_hb=False):
    residues = parse_structure(pdb, chain)
    if not residues:
        raise SystemExit(f"No residues parsed on chain {chain}")
    ensure_H_coords(residues)
    HB = find_hbonds(residues, cutoff=cutoff)
    L = len(residues)

    G, H, I, t3, t4, t5 = assign_helices_all(HB, L)
    bridges = find_bridges(HB, L)
    stateB, stateE, ladders = assign_beta_states(bridges, L)

    temp = merge_states(H, stateB, stateE, G, I)
    T = assign_T_from_turns(t3, t4, t5, temp)

    FINAL = merge_states(H, stateB, stateE, G, I, T)

    s, idxline, aaline = format_summary(residues, FINAL)
    print(f"# File: {pdb}  Chain: {chain}  Cutoff: {cutoff} kcal/mol")
    print(f"# Residues parsed: {L}")
    print("# Summary (per residue):")
    print(s)
    print("# PDB resseq (for reference):")
    print(idxline)
    if verbose:
        print("# Resnames (3-letter):")
        print(aaline)
        unique, counts = np.unique(FINAL, return_counts=True)
        stats = dict(zip(unique, counts))
        print("# Counts:", " ".join(f"{k}:{v}" for k, v in sorted(stats.items(), key=lambda x: 
-PRIORITY.get(x[0],0))))

    if print_hb:
        pairs = sorted(HB.items(), key=lambda x: x[1])[:20]  # show 20 strongest H-bonds
        print("# Example H-bonds CO(i)->NH(j) (PDB resseq)  E(kcal/mol)")
        for (i, j), E in pairs:
            ri = residues[i]["resseq"]; rj = residues[j]["resseq"]
            print(f"  H({ri},{rj})  E={E:.2f}")

    if print_bridges:
        print("# Bridges detected (PDB resseq)  type=P/A")
        for (i, j, t) in bridges:
            ri = residues[i]["resseq"]; rj = residues[j]["resseq"]
            print(f"  ({ri},{rj})  {t}")

    return FINAL, residues, HB, bridges, ladders

# ---------- CLI ----------
if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Mini-DSSP (H/G/I, E/B, T)")
    ap.add_argument("pdb", help="Path to PDB file")
    ap.add_argument("--chain", default="A", help="Chain ID (default A)")
    ap.add_argument("--cutoff", type=float, default=-0.45,
                    help="H-bond energy cutoff in kcal/mol (default -0.45)")
    ap.add_argument("-v", "--verbose", action="store_true",
                    help="Verbose: AA names and counts")
    ap.add_argument("--print-bridges", action="store_true",
                    help="Print detected beta-bridges (P/A)")
    ap.add_argument("--print-hb", action="store_true",
                    help="Print top hydrogen bonds (PDB numbering)")
    args = ap.parse_args()
    run(args.pdb, args.chain, args.cutoff, verbose=args.verbose,
        print_bridges=args.print_bridges, print_hb=args.print_hb)

