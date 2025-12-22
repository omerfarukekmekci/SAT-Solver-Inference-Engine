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

from typing import List, Optional, Tuple


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
        
        # Track if DL0 initial scan has been completed
        self.dl0_done = False

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
# DIMACS parser (Fixed for robustness)
# -----------------------------

def parse_dimacs_cnf(path: str) -> CNFState:
    num_vars = 0
    max_var_seen = 0
    
    with open(path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    clauses: List[Clause] = []
    current_lits: List[int] = []

    line_i = 0
    while line_i < len(lines):
        line = lines[line_i].strip()
        line_i += 1
        if len(line) == 0 or line.startswith("c"):
            continue
        if line.startswith("p"):
            parts = line.split()
            if len(parts) >= 4 and parts[1] == "cnf":
                num_vars = int(parts[2])
            continue

        parts = line.split()
        pi = 0
        while pi < len(parts):
            lit = int(parts[pi])
            pi += 1
            if lit == 0:
                cid = len(clauses) + 1
                # Copy manually
                tmp = []
                for x in current_lits: tmp.append(x)
                clauses.append(Clause(cid, tmp))
                current_lits = []
            else:
                current_lits.append(lit)
                # Robustness update: track actual max var
                v = abs_int(lit)
                if v > max_var_seen:
                    max_var_seen = v

    # FIX: Handle trailing clause without zero
    if len(current_lits) > 0:
        cid = len(clauses) + 1
        tmp = []
        for x in current_lits: tmp.append(x)
        clauses.append(Clause(cid, tmp))

    # Use larger of header vs seen to avoid crash
    final_num_vars = num_vars if num_vars > max_var_seen else max_var_seen
    
    state = CNFState(final_num_vars, clauses)
    state.init_2wl()
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
            self._log(res, dl, "DECIDE      L=" + str(lit).ljust(5) + "|")
        else:
            # forced assignment (deduction)
            self._log(res, dl, "ASSIGN      L=" + str(lit).ljust(5) + "|")
            if reason_clause is not None:
                res.new_deductions.append((lit, reason_clause))
        
        # Log clauses satisfied by this literal
        lit_idx = lit_to_index(lit, self.state.num_vars)
        watch_bucket = self.state.watch_list[lit_idx]
        for cid in watch_bucket:
            self._log(res, dl, "SATISFIED         | C" + str(cid))
        
        return True

    def _process_watched_false_literal(self, res: BCPResult, dl: int, false_lit: int, queue: List[int]) -> bool:
        # When literal becomes FALSE, inspect clauses watching that literal (via watch list).
        num_vars = self.state.num_vars
        wl_idx = lit_to_index(false_lit, num_vars)
        watch_bucket = self.state.watch_list[wl_idx]

        i = 0
        while i < len(watch_bucket):
            cid = watch_bucket[i]
            clause = self.state.get_clause_by_id(cid)

            wpos = -1
            other_pos = -1

            if clause.w1 != -1 and clause.lits[clause.w1] == false_lit:
                wpos = clause.w1
                other_pos = clause.w2
            elif clause.w2 != -1 and clause.lits[clause.w2] == false_lit:
                wpos = clause.w2
                other_pos = clause.w1
            else:
                i += 1
                continue

            if other_pos == -1:
                # Unit clause handling
                if lit_is_false(clause.lits[wpos], self.state.assignment):
                    res.status = "CONFLICT"
                    res.conflict_id = cid
                    self._log(res, dl, "CONFLICT          | Violation: C" + str(cid))
                    return False
                i += 1
                continue

            other_lit = clause.lits[other_pos]

            if lit_is_true(other_lit, self.state.assignment):
                self._log(res, dl, "SATISFIED         | C" + str(cid))
                i += 1
                continue

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
                if clause.w1 == wpos:
                    clause.w1 = rep_pos
                else:
                    clause.w2 = rep_pos

                last = watch_bucket[len(watch_bucket) - 1]
                watch_bucket[i] = last
                watch_bucket.pop()

                new_idx = lit_to_index(new_lit, num_vars)
                self.state.watch_list[new_idx].append(cid)

                # FIX: Removed abs_int from log to show true sign
                self._log(
                    res,
                    dl,
                    "SHIFT L=" + str(false_lit) + " | C" + str(cid) + " " +
                    str(false_lit) + "->" + str(new_lit)
                )
                continue

            if lit_is_unassigned(other_lit, self.state.assignment):
                self._log(res, dl, "UNIT        L=" + str(other_lit).ljust(5) + "| C" + str(cid))

                ok = self._set_literal(res, dl, other_lit, cid, is_decision=False)
                if not ok:
                    res.status = "CONFLICT"
                    res.conflict_id = cid
                    self._log(res, dl, "CONFLICT          | Violation: C" + str(cid))
                    return False

                self._enqueue(queue, -other_lit)
                self._log(res, dl, "SATISFIED         | C" + str(cid))
                i += 1
                continue

            if lit_is_false(other_lit, self.state.assignment):
                res.status = "CONFLICT"
                res.conflict_id = cid
                self._log(res, dl, "CONFLICT          | Violation: C" + str(cid))
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
        queue: List[int] = []
        dl = 0
        res.dl = 0

        ci = 0
        while ci < len(self.state.clauses):
            c = self.state.clauses[ci]
            if len(c.lits) == 1:
                lit = c.lits[0]
                v = abs_int(lit)
                cur = self.state.assignment[v]
                desired = 1 if lit > 0 else -1
                if cur != 0 and cur != desired:
                    res.status = "CONFLICT"
                    res.conflict_id = c.cid
                    self._log(res, dl, "CONFLICT          | Violation: C" + str(c.cid))
                    return False

                if cur == 0:
                    self._log(res, dl, "UNIT        L=" + str(lit).ljust(5) + "| C" + str(c.cid))
                    ok = self._set_literal(res, dl, lit, c.cid, is_decision=False)
                    if not ok:
                        res.status = "CONFLICT"
                        res.conflict_id = c.cid
                        self._log(res, dl, "CONFLICT          | Violation: C" + str(c.cid))
                        return False
                    self._log(res, dl, "SATISFIED         | C" + str(c.cid))
                    self._enqueue(queue, -lit)
            ci += 1

        # Process queue in LIFO order (stack-based) to match expected output
        while len(queue) > 0:
            falselit = queue.pop()  # Pop from end (LIFO)
            ok = self._process_watched_false_literal(res, dl, falselit, queue)
            if not ok:
                return False

        if self._all_assigned():
            res.status = "SAT"
        else:
            res.status = "CONTINUE"
        return True

    def run_bcp(self, trigger_lit: Optional[int], dl: int) -> BCPResult:
        res = BCPResult()
        
        # Only run DL0 initial scan once per instance
        if not self.state.dl0_done:
            ok0 = self._initial_unit_scan_dl0(res)
            self.state.dl0_done = True
            
            if not ok0:
                res.dl = 0
                return res

            if res.status == "SAT":
                res.dl = 0
                return res

        if trigger_lit is None or trigger_lit == 0:
            res.dl = 0 if not self.state.dl0_done else dl
            return res

        res.dl = dl

        ok_dec = self._set_literal(res, dl, trigger_lit, None, is_decision=True)
        if not ok_dec:
            res.status = "CONFLICT"
            res.conflict_id = None
            self._log(res, dl, "CONFLICT | Violation: None")
            return res

        queue: List[int] = []
        self._enqueue(queue, -trigger_lit)

        # Process queue in LIFO order (stack-based) to match expected output
        while len(queue) > 0:
            falselit = queue.pop()  # Pop from end (LIFO)
            ok = self._process_watched_false_literal(res, dl, falselit, queue)
            if not ok:
                return res

        if self._all_assigned():
            res.status = "SAT"
        else:
            res.status = "CONTINUE"

        return res

    def apply_previous_assignments(self, literals: List[int]) -> bool:
        # Used to inject state from previous runs without logging it
        res = BCPResult() 
        dl = 0
        queue: List[int] = []
        
        for lit in literals:
            if lit_is_false(lit, self.state.assignment):
                return False 
            if lit_is_true(lit, self.state.assignment):
                continue
                
            ok = self._set_literal(res, dl, lit, None, is_decision=False)
            if not ok:
                return False
            self._enqueue(queue, -lit)
            
        # Process queue in LIFO order (stack-based)
        while len(queue) > 0:
            falselit = queue.pop()  # Pop from end (LIFO)
            ok = self._process_watched_false_literal(res, dl, falselit, queue)
            if not ok:
                return False
        return True


# -----------------------------
# I/O 
# -----------------------------

def read_trigger_file(path: str) -> Tuple[int, int]:
    # Standard format again (no PREVIOUS_ASSIGNS needed in file)
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
        # Convert CONFLICT to UNSAT for output
        status_output = "UNSAT" if res.status == "CONFLICT" else res.status
        f.write("STATUS: " + status_output + "\n")
        f.write("DL: " + str(res.dl) + "\n")
        if res.conflict_id is None:
            f.write("CONFLICT_ID: None\n")
        else:
            f.write("CONFLICT_ID: " + str(res.conflict_id) + "\n")

        f.write("\n")  # Empty line before BCP log
        f.write("--- BCP EXECUTION LOG ---\n")
        li = 0
        while li < len(res.exec_log):
            f.write(res.exec_log[li] + "\n")
            li += 1
        
        # Add BACKTRACK line if there was a conflict
        if res.status == "CONFLICT":
            f.write("[DL" + str(res.dl) + "] BACKTRACK         |\n")
        
        f.write("\n")  # Empty line before variable state
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
            f.write(str(v).ljust(4) + " | " + s + "\n")
            v += 1


# -----------------------------
# Sequential Decision Loading Helpers
# -----------------------------

def get_sample_info(trigger_path: str) -> Tuple[int, str]:
    # Extract sample number and base directory from trigger file path
    # Example: "SampleRuns\\4\\decision2_4.txt" -> (4, "SampleRuns\\4")
    # Example: "SampleRuns/4/decision2_4.txt" -> (4, "SampleRuns/4")
    
    import os
    import re
    
    # Normalize path separators
    norm_path = trigger_path.replace("\\", "/")
    
    # Extract filename
    filename = os.path.basename(norm_path)
    
    # Extract sample number from filename (e.g., "decision2_4.txt" -> 4)
    match = re.search(r'_(\d+)\.txt$', filename)
    if match:
        sample_num = int(match.group(1))
    else:
        sample_num = 0
    
    # Get base directory
    base_dir = os.path.dirname(trigger_path)
    
    return sample_num, base_dir

def find_previous_decisions(trigger_path: str, current_dl: int, sample_num: int, base_dir: str) -> List[str]:
    # Find all previous decision files for DL levels 1 to current_dl-1
    # Returns list like ["SampleRuns/4/decision1_4.txt"] for DL=2
    
    import os
    
    prev_decisions = []
    
    dl = 1
    while dl < current_dl:
        # Construct filename: decision<DL>_<sample_num>.txt
        filename = "decision" + str(dl) + "_" + str(sample_num) + ".txt"
        decision_path = os.path.join(base_dir, filename)
        
        # Check if file exists
        if os.path.exists(decision_path):
            prev_decisions.append(decision_path)
        
        dl += 1
    
    return prev_decisions

def run_sequential_decisions(state: CNFState, engine: InferenceEngine, 
                             cnf_path: str, decision_paths: List[str]) -> List[int]:
    # Run each decision file in order and accumulate assigned literals
    # Returns list of all literals assigned across all previous decisions
    
    all_assigned = []
    
    i = 0
    while i < len(decision_paths):
        dec_path = decision_paths[i]
        
        # Read trigger from this decision file
        trig, dl = read_trigger_file(dec_path)
        
        # Run BCP for this decision
        res = engine.run_bcp(trig, dl)
        
        # If conflict, we can't continue
        if res.status == "CONFLICT":
            return all_assigned
        
        # Collect newly assigned literals from this decision
        # We track the trigger literal itself
        if trig != 0:
            all_assigned.append(trig)
        
        i += 1
    
    return all_assigned

# -----------------------------
# Main
# -----------------------------

def main():
    import sys
    if len(sys.argv) < 3:
        print("Usage: python main.py <problem.cnf> [<trigger.txt>] <bcp_output.txt>")
        print("  2 args: python main.py <problem.cnf> <output.txt>  (DL0 mode)")
        print("  3 args: python main.py <problem.cnf> <trigger.txt> <output.txt>")
        return

    cnf_path = sys.argv[1]
    
    # Support both 2-arg (DL0) and 3-arg (with trigger) modes
    if len(sys.argv) == 3:
        # DL0 mode: no trigger file
        trigger_path = None
        out_path = sys.argv[2]
        trig = 0
        dl = 0
    else:
        # Normal mode: with trigger file
        trigger_path = sys.argv[2]
        out_path = sys.argv[3]
        trig, dl = read_trigger_file(trigger_path)

    state = parse_dimacs_cnf(cnf_path)
    engine = InferenceEngine(state)
    
    # Auto-load previous decisions if DL > 1
    if dl > 1 and trigger_path is not None:
        sample_num, base_dir = get_sample_info(trigger_path)
        prev_decision_paths = find_previous_decisions(trigger_path, dl, sample_num, base_dir)
        
        if len(prev_decision_paths) > 0:
            # Run all previous decisions to build up state
            prev_assigned = run_sequential_decisions(state, engine, cnf_path, prev_decision_paths)

    res = engine.run_bcp(trig, dl)

    write_bcp_output(out_path, res, state)

if __name__ == "__main__":
    main()