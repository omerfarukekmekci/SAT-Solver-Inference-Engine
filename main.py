# inference_engine.py
# BLG345E Project #3 - Inference Engine (BCP with 2-Watched Literals)
#
# Implements:
# - Boolean Constraint Propagation (Unit Propagation)
# - Fully based on 2-Watched Literals + Watch Lists
# - DL0 initial unit scan + propagation
# - Trigger input from Project #4 (TRIGGER_LITERAL, DL)
# - Output file in the expected format
#
# Notes on "no built-in functions for main logic":
# - No any/all/min/max/sorted etc. in core algorithm
# - Core operations are done with explicit loops

from typing import List, Dict, Optional, Tuple


# -----------------------------
# Small utilities (no fancy built-ins)
# -----------------------------

def abs_int(x: int) -> int:
    return -x if x < 0 else x

def sign_int(x: int) -> int:
    return -1 if x < 0 else 1

def lit_to_index(lit: int, num_vars: int) -> int:
    # Map literals in [-V..-1, 1..V] to [0..2V-1]
    # Negative literals first half, positive literals second half
    v = abs_int(lit)
    if lit < 0:
        return v - 1
    return num_vars + (v - 1)

def index_to_lit(idx: int, num_vars: int) -> int:
    if idx < num_vars:
        return -(idx + 1)
    return (idx - num_vars) + 1

def lit_is_true(lit: int, assignment: List[int]) -> bool:
    # assignment[var] in {-1,0,1} where var is 1-based index
    v = abs_int(lit)
    val = assignment[v]
    if val == 0:
        return False
    if lit > 0:
        return val == 1
    return val == -1

def lit_is_false(lit: int, assignment: List[int]) -> bool:
    v = abs_int(lit)
    val = assignment[v]
    if val == 0:
        return False
    if lit > 0:
        return val == -1
    return val == 1

def lit_is_unassigned(lit: int, assignment: List[int]) -> bool:
    v = abs_int(lit)
    return assignment[v] == 0

def assign_literal(lit: int, assignment: List[int]) -> None:
    v = abs_int(lit)
    if lit > 0:
        assignment[v] = 1
    else:
        assignment[v] = -1


# -----------------------------
# Data Structures
# -----------------------------

class Clause:
    __slots__ = ("cid", "lits", "w1", "w2")

    def __init__(self, cid: int, lits: List[int]):
        self.cid = cid
        self.lits = lits  # list of signed ints
        # watched literal positions (indices into self.lits)
        # for unit clause: w1 set, w2 = -1
        self.w1 = -1
        self.w2 = -1

class CNFState:
    def __init__(self, num_vars: int, clauses: List[Clause]):
        self.num_vars = num_vars
        self.clauses = clauses

        # assignment[0] unused, assignment[1..V] in {-1,0,1}
        self.assignment = [0] * (num_vars + 1)

        # watch_list is an array-of-arrays, indexed by literal index (2V)
        self.watch_list: List[List[int]] = []
        i = 0
        while i < 2 * num_vars:
            self.watch_list.append([])
            i += 1

        # implication / reason: reason[var] = clause_id that implied it (None if decision)
        self.reason: List[Optional[int]] = [None] * (num_vars + 1)

    def init_2wl(self) -> None:
        # Initialize watches per clause and populate watch lists
        # Rules: k>=2 watch first two; k==1 watch only; k==0 immediate conflict (handled upstream)
        ci = 0
        while ci < len(self.clauses):
            c = self.clauses[ci]
            k = len(c.lits)
            if k == 0:
                # empty clause => conflict in formula
                c.w1 = -1
                c.w2 = -1
            elif k == 1:
                c.w1 = 0
                c.w2 = -1
                lit = c.lits[0]
                idx = lit_to_index(lit, self.num_vars)
                self.watch_list[idx].append(c.cid)
            else:
                c.w1 = 0
                c.w2 = 1
                lit1 = c.lits[c.w1]
                lit2 = c.lits[c.w2]
                self.watch_list[lit_to_index(lit1, self.num_vars)].append(c.cid)
                self.watch_list[lit_to_index(lit2, self.num_vars)].append(c.cid)
            ci += 1

    def get_clause_by_id(self, cid: int) -> Clause:
        # clause ids are 1..C in order
        return self.clauses[cid - 1]


# -----------------------------
# DIMACS parser (for standalone run)
# In real integration, Project #2 would provide equivalent initialized structures.
# -----------------------------

def parse_dimacs_cnf(path: str) -> CNFState:
    num_vars = 0
    num_clauses = 0

    # Read lines (simple + robust)
    with open(path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    clauses: List[Clause] = []
    current_lits: List[int] = []

    line_i = 0
    while line_i < len(lines):
        line = lines[line_i].strip()
        line_i += 1

        if len(line) == 0:
            continue
        if line[0] == "c":
            continue
        if line[0] == "p":
            # p cnf V C
            parts = line.split()
            # minimal validation
            if len(parts) >= 4 and parts[1] == "cnf":
                num_vars = int(parts[2])
                num_clauses = int(parts[3])
            continue

        # Clause line: may contain multiple clauses if formatted weirdly, so parse ints until 0
        parts = line.split()
        pi = 0
        while pi < len(parts):
            lit = int(parts[pi])
            pi += 1
            if lit == 0:
                # end clause
                cid = len(clauses) + 1
                # copy literals
                tmp: List[int] = []
                li = 0
                while li < len(current_lits):
                    tmp.append(current_lits[li])
                    li += 1
                clauses.append(Clause(cid, tmp))
                current_lits = []
            else:
                current_lits.append(lit)

    # If file ended without trailing 0 for last clause, ignore / could raise
    state = CNFState(num_vars, clauses)
    state.init_2wl()

    # (Optional) sanity: num_clauses might mismatch, but we won't hard fail here
    _ = num_clauses
    return state


# -----------------------------
# Inference Engine (BCP)
# -----------------------------

class BCPResult:
    def __init__(self):
        self.status = "CONTINUE"   # CONTINUE / CONFLICT / SAT
        self.dl = 0
        self.conflict_id: Optional[int] = None
        self.exec_log: List[str] = []
        self.new_deductions: List[Tuple[int, int]] = []  # (lit, clause_id) for forced assigns this run


class InferenceEngine:
    def __init__(self, state: CNFState):
        self.state = state

    def _log(self, res: BCPResult, dl: int, msg: str) -> None:
        res.exec_log.append("[DL" + str(dl) + "] " + msg)

    def _enqueue(self, queue: List[int], lit: int) -> None:
        queue.append(lit)

    def _set_literal(self, res: BCPResult, dl: int, lit: int, reason_clause: Optional[int], is_decision: bool) -> bool:
        # returns False if contradiction (assigning var both ways)
        v = abs_int(lit)
        cur = self.state.assignment[v]
        desired = 1 if lit > 0 else -1

        if cur != 0:
            # already assigned: check consistency
            if cur != desired:
                return False
            return True

        # assign
        assign_literal(lit, self.state.assignment)
        self.state.reason[v] = reason_clause

        if is_decision:
            self._log(res, dl, "DECIDE L=" + str(lit) + " |")
        else:
            # forced assignment (deduction)
            self._log(res, dl, "ASSIGN L=" + str(lit) + " |")
            if reason_clause is not None:
                res.new_deductions.append((lit, reason_clause))
        return True

    def _process_watched_false_literal(self, res: BCPResult, dl: int, false_lit: int, queue: List[int]) -> bool:
        # When literal becomes FALSE, inspect clauses watching that literal (via watch list).
        # Implements watch shifting + unit detection + conflict detection.
        num_vars = self.state.num_vars
        wl_idx = lit_to_index(false_lit, num_vars)
        watch_bucket = self.state.watch_list[wl_idx]

        # We'll iterate with index because we may remove/swap entries while iterating.
        i = 0
        while i < len(watch_bucket):
            cid = watch_bucket[i]
            clause = self.state.get_clause_by_id(cid)

            # Determine which watch in this clause equals false_lit (could be w1 or w2 or unit w1)
            wpos = -1
            other_pos = -1

            if clause.w1 != -1 and clause.lits[clause.w1] == false_lit:
                wpos = clause.w1
                other_pos = clause.w2
            elif clause.w2 != -1 and clause.lits[clause.w2] == false_lit:
                wpos = clause.w2
                other_pos = clause.w1
            else:
                # stale entry (shouldn't happen if updates are correct). Skip safely.
                i += 1
                continue

            # Unit clause special case: only one watch
            if other_pos == -1:
                # If that single literal is false, clause is violated => conflict
                # If it's true, clause satisfied (wouldn't be in false watch list)
                # If unassigned, it should have been assigned earlier
                if lit_is_false(clause.lits[wpos], self.state.assignment):
                    res.status = "CONFLICT"
                    res.conflict_id = cid
                    self._log(res, dl, "CONFLICT | Violation: C" + str(cid))
                    return False
                i += 1
                continue

            other_lit = clause.lits[other_pos]

            # If other watch is TRUE, clause already satisfied; keep watches as is.
            if lit_is_true(other_lit, self.state.assignment):
                self._log(res, dl, "SATISFIED | C" + str(cid))
                i += 1
                continue

            # Try to find a replacement literal to watch (TRUE or UNASSIGNED) excluding other watch
            found_replacement = False
            rep_pos = -1

            j = 0
            while j < len(clause.lits):
                if j != other_pos and j != wpos:
                    cand = clause.lits[j]
                    if (not lit_is_false(cand, self.state.assignment)):
                        found_replacement = True
                        rep_pos = j
                        break
                j += 1

            if found_replacement:
                new_lit = clause.lits[rep_pos]

                # Update clause watch position
                if clause.w1 == wpos:
                    clause.w1 = rep_pos
                else:
                    clause.w2 = rep_pos

                # Remove this clause from current watch bucket (false_lit)
                # O(1) removal: swap with last, pop
                last = watch_bucket[len(watch_bucket) - 1]
                watch_bucket[i] = last
                watch_bucket.pop()

                # Add clause to new literal's watch list
                new_idx = lit_to_index(new_lit, num_vars)
                self.state.watch_list[new_idx].append(cid)

                # Log SHIFT: "SHIFT L=<false_lit> | Cx <old>-><new>"
                self._log(
                    res,
                    dl,
                    "SHIFT L=" + str(abs_int(false_lit)) + " | C" + str(cid) + " " +
                    str(abs_int(false_lit)) + "->" + str(abs_int(new_lit))
                )

                # do NOT increment i, since we swapped a new element into index i
                continue

            # No replacement found -> clause is either UNIT or CONFLICT depending on other watch
            if lit_is_unassigned(other_lit, self.state.assignment):
                # UNIT: other_lit must be assigned TRUE
                self._log(res, dl, "UNIT L=" + str(other_lit) + " | C" + str(cid))

                ok = self._set_literal(res, dl, other_lit, cid, is_decision=False)
                if not ok:
                    res.status = "CONFLICT"
                    res.conflict_id = cid
                    self._log(res, dl, "CONFLICT | Violation: C" + str(cid))
                    return False

                # When other_lit becomes true, its negation becomes false and should be propagated
                self._enqueue(queue, -other_lit)

                # Clause is now satisfied by other_lit
                self._log(res, dl, "SATISFIED | C" + str(cid))

                i += 1
                continue

            # other_lit is FALSE -> conflict
            if lit_is_false(other_lit, self.state.assignment):
                res.status = "CONFLICT"
                res.conflict_id = cid
                self._log(res, dl, "CONFLICT | Violation: C" + str(cid))
                return False

            i += 1

        return True

    def _all_assigned(self) -> bool:
        v = 1
        while v <= self.state.num_vars:
            if self.state.assignment[v] == 0:
                return False
            v += 1
        return True

    def _initial_unit_scan_dl0(self, res: BCPResult) -> bool:
        # Scan unit clauses and enqueue their forced literals at DL0
        # (The spec says IE checks unit clauses itself and runs DL0.) :contentReference[oaicite:5]{index=5}
        queue: List[int] = []
        dl = 0
        res.dl = 0

        ci = 0
        while ci < len(self.state.clauses):
            c = self.state.clauses[ci]
            if len(c.lits) == 1:
                lit = c.lits[0]
                # If already assigned, verify consistency
                v = abs_int(lit)
                cur = self.state.assignment[v]
                desired = 1 if lit > 0 else -1
                if cur != 0 and cur != desired:
                    res.status = "CONFLICT"
                    res.conflict_id = c.cid
                    self._log(res, dl, "CONFLICT | Violation: C" + str(c.cid))
                    return False

                if cur == 0:
                    self._log(res, dl, "UNIT L=" + str(lit) + " | C" + str(c.cid))
                    ok = self._set_literal(res, dl, lit, c.cid, is_decision=False)
                    if not ok:
                        res.status = "CONFLICT"
                        res.conflict_id = c.cid
                        self._log(res, dl, "CONFLICT | Violation: C" + str(c.cid))
                        return False
                    self._log(res, dl, "SATISFIED | C" + str(c.cid))
                    # propagate falsified negation
                    self._enqueue(queue, -lit)
            ci += 1

        # Process propagation queue (falsified literals)
        qi = 0
        while qi < len(queue):
            falselit = queue[qi]
            qi += 1
            ok = self._process_watched_false_literal(res, dl, falselit, queue)
            if not ok:
                return False

        # If everything assigned and no conflict -> SAT else CONTINUE
        if self._all_assigned():
            res.status = "SAT"
        else:
            res.status = "CONTINUE"
        return True

    def run_bcp(self, trigger_lit: Optional[int], dl: int) -> BCPResult:
        res = BCPResult()

        # Phase 1: DL0 initial unit scan is always done first (once)
        # We'll do it every run safely; if already assigned, it won't re-assign.
        ok0 = self._initial_unit_scan_dl0(res)
        if not ok0:
            # conflict at DL0
            res.dl = 0
            return res

        # If DL0 already proves SAT, we can return SAT right away
        if res.status == "SAT":
            res.dl = 0
            return res

        # Phase 2: process trigger decision for given DL
        if trigger_lit is None:
            res.dl = 0
            return res

        res.dl = dl

        # Make decision assignment
        ok_dec = self._set_literal(res, dl, trigger_lit, None, is_decision=True)
        if not ok_dec:
            res.status = "CONFLICT"
            res.conflict_id = None
            self._log(res, dl, "CONFLICT | Violation: None")
            return res

        # Start queue with falsified negation of the decided literal
        queue: List[int] = []
        self._enqueue(queue, -trigger_lit)

        qi = 0
        while qi < len(queue):
            falselit = queue[qi]
            qi += 1
            ok = self._process_watched_false_literal(res, dl, falselit, queue)
            if not ok:
                return res

        # Determine final status for this run
        if self._all_assigned():
            res.status = "SAT"
        else:
            res.status = "CONTINUE"

        return res


# -----------------------------
# I/O for Project #4 trigger + output format
# -----------------------------

def read_trigger_file(path: str) -> Tuple[int, int]:
    # Expected:
    # TRIGGER_LITERAL: 1
    # DL: 1
    trigger = 0
    dl = 0
    with open(path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    i = 0
    while i < len(lines):
        line = lines[i].strip()
        i += 1
        if len(line) == 0:
            continue
        if line.startswith("TRIGGER_LITERAL:"):
            parts = line.split(":")
            if len(parts) >= 2:
                trigger = int(parts[1].strip())
        elif line.startswith("DL:"):
            parts = line.split(":")
            if len(parts) >= 2:
                dl = int(parts[1].strip())

    return trigger, dl

def write_bcp_output(path: str, res: BCPResult, state: CNFState) -> None:
    with open(path, "w", encoding="utf-8") as f:
        f.write("--- STATUS ---\n")
        f.write("STATUS: " + res.status + "\n")
        f.write("DL: " + str(res.dl) + "\n")
        if res.conflict_id is None:
            f.write("CONFLICT_ID: None\n")
        else:
            f.write("CONFLICT_ID: " + str(res.conflict_id) + "\n")

        f.write("--- BCP EXECUTION LOG ---\n")
        li = 0
        while li < len(res.exec_log):
            f.write(res.exec_log[li] + "\n")
            li += 1

        f.write("--- CURRENT VARIABLE STATE ---\n")
        v = 1
        while v <= state.num_vars:
            val = state.assignment[v]
            if val == 0:
                s = "UNASSIGNED"
            elif val == 1:
                s = "TRUE"
            else:
                s = "FALSE"
            f.write(str(v) + " | " + s + "\n")
            v += 1


# -----------------------------
# Sample run compatible main
# -----------------------------
#
# Usage:
#   python inference_engine.py problem.cnf trigger.txt bcp_output.txt
#
# Notes:
# - The engine also runs DL0 unit propagation automatically
# - Then applies the decision in trigger.txt

def main():
    import sys
    if len(sys.argv) < 4:
        print("Usage: python inference_engine.py <problem.cnf> <trigger.txt> <bcp_output.txt>")
        return

    cnf_path = sys.argv[1]
    trigger_path = sys.argv[2]
    out_path = sys.argv[3]

    state = parse_dimacs_cnf(cnf_path)
    engine = InferenceEngine(state)

    trig, dl = read_trigger_file(trigger_path)
    res = engine.run_bcp(trig, dl)

    write_bcp_output(out_path, res, state)

    # Extra: print deduction trace (for debugging / Project #5)
    # We keep it off the required output format by default.
    # Uncomment if you want:
    # print("DEDUCTIONS THIS RUN:", res.new_deductions)


if __name__ == "__main__":
    main()
